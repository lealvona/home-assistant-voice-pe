#pragma once
//
// voice_dispatch.h — non-blocking HTTP dispatch helpers for the lealvona
// Home Assistant Voice PE fork.
//
// The custom long-press (research) and single-press / wake-word (quick chat)
// flows POST a transcript to LAN endpoints. Doing that synchronously on the
// ESPHome main loop stalls the loop (and the LED animations) for the duration
// of the request — up to ~3.8s for the research bridge+fallback path.
//
// This header moves each POST onto a short-lived FreeRTOS task so the main
// loop never blocks. The worker tasks deliberately touch ONLY esp-idf HTTP and
// std::atomic — they never call back into ESPHome components, scripts or the
// scheduler (which are single-threaded and not safe to touch off the main
// loop). Results the UI cares about (a quick-chat POST failure) are exposed
// through an atomic that the main loop polls, so all ESPHome state changes
// still happen on the main loop.
//
#include <atomic>
#include <cstdio>
#include <string>
#include <utility>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_http_client.h"
#include "esp_log.h"

namespace voice_dispatch {

// Cap on concurrent in-flight POST tasks, so a mashed button (or a wedged
// network) can't spawn unbounded tasks and exhaust internal RAM.
constexpr int MAX_INFLIGHT = 4;
// Worker stack in BYTES (esp-idf's xTaskCreate uses bytes, not words). Sized
// with headroom for esp_http_client's request/response buffering on top of the
// heap-copied body and log formatting.
constexpr uint32_t TASK_STACK = 12288;
constexpr UBaseType_t TASK_PRIO = 5;

inline std::atomic<int> &inflight() {
  static std::atomic<int> n{0};
  return n;
}

// Quick-chat POST result, packed as (generation << 2) | code, observed on the
// main loop. code: 0 = pending, 1 = ok, 2 = failed, 3 = clarify (the bridge
// needs a spoken local/cloud answer before it can dispatch). The generation tag
// lets the main loop ignore a stale worker's result from a press that has since
// been superseded (mirrors the quick_chat_active_generation guard used elsewhere).
inline std::atomic<int> &qc_state() {
  static std::atomic<int> s{(-1 << 2)};  // generation -1, code 0 -> never a failure
  return s;
}

// If a quick-chat POST has just failed, returns the generation it belonged to
// (and clears the slot so it reports once); otherwise returns -1. The caller
// compares the generation against the live quick_chat_generation before acting,
// so a superseded press cannot disturb the current session.
inline int take_quickchat_failure_gen() {
  int s = qc_state().load();
  if ((s & 0x3) == 2) {
    int gen = s >> 2;
    int cleared = gen << 2;  // back to "pending" for this generation
    qc_state().compare_exchange_strong(s, cleared);
    return gen;
  }
  return -1;
}

// Non-destructive peek: returns the generation of a pending clarify (code 3)
// without clearing it, else -1. Lets the poll interval check it can actually
// re-arm STT BEFORE consuming the signal, so a momentary not-ready window
// (VA still stopping, muted, API blip) can't strand the clarification with the
// signal already gone.
inline int peek_quickchat_clarify_gen() {
  int s = qc_state().load();
  return ((s & 0x3) == 3) ? (s >> 2) : -1;
}

// If the bridge asked for a local/cloud clarification (code 3), returns the
// generation it belonged to (and clears the slot so it reports once); otherwise
// -1. Mirrors take_quickchat_failure_gen so the main loop's poll interval can
// re-arm STT for the follow-up answer without disturbing a superseded press.
inline int take_quickchat_clarify_gen() {
  int s = qc_state().load();
  if ((s & 0x3) == 3) {
    int gen = s >> 2;
    int cleared = gen << 2;  // back to "pending" for this generation
    qc_state().compare_exchange_strong(s, cleared);
    return gen;
  }
  return -1;
}

// JSON-escape a string for embedding inside a double-quoted JSON value.
// Handles the structural characters plus all C0 control characters (so an
// unexpected control byte in a transcript can't produce invalid JSON).
inline std::string json_escape(const std::string &in) {
  std::string out;
  out.reserve(in.size() + 16);
  for (char c : in) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[7];
          snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

// Perform one blocking JSON POST. Runs on a worker task, never the main loop.
// Returns the HTTP status code on a 2xx response, or a negative value on any
// transport error / non-2xx. Logs tag + status only (never the URL, to keep
// internal topology out of device logs).
inline int do_post(const std::string &url, const std::string &body, uint32_t timeout_ms, const char *tag) {
  esp_http_client_config_t cfg = {};
  cfg.url = url.c_str();
  cfg.method = HTTP_METHOD_POST;
  cfg.timeout_ms = static_cast<int>(timeout_ms);
  esp_http_client_handle_t client = esp_http_client_init(&cfg);
  if (client == nullptr) {
    ESP_LOGW(tag, "failed to init HTTP client");
    return -1;
  }
  esp_http_client_set_header(client, "Content-Type", "application/json");
  esp_http_client_set_post_field(client, body.c_str(), body.length());
  esp_err_t err = esp_http_client_perform(client);
  int status = esp_http_client_get_status_code(client);
  ESP_LOGI(tag, "POST: %s http_status=%d", esp_err_to_name(err), status);
  esp_http_client_cleanup(client);
  bool ok = (err == ESP_OK) && (status >= 200) && (status < 300);
  return ok ? status : -1;
}

// Like do_post, but captures up to 255 bytes of the response body into
// `resp_out` (used by quick-chat to detect the bridge's {"status":"clarify"}
// signal). Uses the manual open/write/fetch/read sequence because
// esp_http_client_perform() discards the body. Returns the 2xx status, or a
// negative value on transport error / non-2xx.
inline int do_post_capture(const std::string &url, const std::string &body, uint32_t timeout_ms,
                           const char *tag, std::string &resp_out) {
  resp_out.clear();
  esp_http_client_config_t cfg = {};
  cfg.url = url.c_str();
  cfg.method = HTTP_METHOD_POST;
  cfg.timeout_ms = static_cast<int>(timeout_ms);
  esp_http_client_handle_t client = esp_http_client_init(&cfg);
  if (client == nullptr) {
    ESP_LOGW(tag, "failed to init HTTP client");
    return -1;
  }
  esp_http_client_set_header(client, "Content-Type", "application/json");
  esp_err_t err = esp_http_client_open(client, body.length());
  if (err != ESP_OK) {
    ESP_LOGW(tag, "open failed: %s", esp_err_to_name(err));
    esp_http_client_cleanup(client);
    return -1;
  }
  int wlen = esp_http_client_write(client, body.data(), body.length());
  if (wlen < 0) {
    ESP_LOGW(tag, "write failed");
    esp_http_client_cleanup(client);
    return -1;
  }
  esp_http_client_fetch_headers(client);
  int status = esp_http_client_get_status_code(client);
  char buf[256];
  int total = 0;
  while (total < static_cast<int>(sizeof(buf)) - 1) {
    int r = esp_http_client_read(client, buf + total, sizeof(buf) - 1 - total);
    if (r <= 0) break;
    total += r;
  }
  resp_out.assign(buf, total > 0 ? total : 0);
  ESP_LOGI(tag, "POST(capture): http_status=%d body_len=%d", status, total);
  esp_http_client_cleanup(client);
  return (status >= 200 && status < 300) ? status : -1;
}

// Spawn a worker task to run `fn(job)`, honouring the in-flight cap. Takes
// ownership of `job`: deletes it (and returns false) if the dispatch is
// dropped or the task can't be created. NOTE: post_*/quickchat_post_async are
// only ever called from the single ESPHome main loop, so the cap's
// load()-then-increment is not a real race; workers only ever decrement.
template<typename Job>
inline bool spawn_worker(const char *name, TaskFunction_t fn, Job *job, const char *tag) {
  if (inflight().load() >= MAX_INFLIGHT) {
    ESP_LOGW(tag, "dropping dispatch: %d worker tasks already in flight", MAX_INFLIGHT);
    delete job;
    return false;
  }
  inflight().fetch_add(1);
  if (xTaskCreate(fn, name, TASK_STACK, job, TASK_PRIO, nullptr) != pdPASS) {
    ESP_LOGE(tag, "failed to spawn worker task");
    inflight().fetch_sub(1);
    delete job;
    return false;
  }
  return true;
}

// ---- fire-and-forget single POST (camera button) ---------------------------

struct PostJob {
  std::string url;
  std::string body;
  uint32_t timeout_ms;
  const char *tag;  // static string literal — safe to hold
};

inline void post_task(void *arg) {
  PostJob *job = static_cast<PostJob *>(arg);
  do_post(job->url, job->body, job->timeout_ms, job->tag);
  delete job;
  inflight().fetch_sub(1);
  vTaskDelete(nullptr);
}

inline bool post_async(std::string url, std::string body, uint32_t timeout_ms, const char *tag) {
  return spawn_worker<PostJob>("va_post", post_task,
                               new PostJob{std::move(url), std::move(body), timeout_ms, tag}, tag);
}

// ---- research dispatch: bridge first, then fallback ------------------------

struct ResearchJob {
  std::string bridge_url;
  std::string fallback_url;
  std::string body;
};

inline void research_task(void *arg) {
  ResearchJob *job = static_cast<ResearchJob *>(arg);
  // Bridge proxy first with a short timeout; on any failure fall back to a
  // direct POST to the research backend so dispatch survives the bridge being
  // down. Both attempts run here, off the main loop.
  int bridge = do_post(job->bridge_url, job->body, 800, "research_webhook");
  if (bridge < 0) {
    ESP_LOGW("research_webhook", "bridge unavailable; falling back to direct research-backend POST");
    do_post(job->fallback_url, job->body, 3000, "research_webhook");
  }
  delete job;
  inflight().fetch_sub(1);
  vTaskDelete(nullptr);
}

inline bool research_async(std::string bridge_url, std::string fallback_url, std::string body) {
  return spawn_worker<ResearchJob>(
      "va_research", research_task,
      new ResearchJob{std::move(bridge_url), std::move(fallback_url), std::move(body)}, "research_webhook");
}

// ---- quick chat: tracked so the main loop can show an error LED ------------

struct QcJob {
  std::string url;
  std::string body;
  uint32_t timeout_ms;
  int generation;
};

inline void qc_task(void *arg) {
  QcJob *job = static_cast<QcJob *>(arg);
  std::string resp;
  int r = do_post_capture(job->url, job->body, job->timeout_ms, "quick_chat", resp);
  // code: 2 = transport/non-2xx failure, 3 = bridge wants a local/cloud
  // clarification (body contains "clarify"), 1 = accepted/processing.
  int code;
  if (r < 0) {
    code = 2;
  } else if (resp.find("clarify") != std::string::npos) {
    code = 3;
  } else {
    code = 1;
  }
  // Only record our result if this generation still owns the slot — if a newer
  // press has taken over, expected won't match and the store is dropped.
  int expected = job->generation << 2;  // "pending" for our generation
  int desired = (job->generation << 2) | code;
  qc_state().compare_exchange_strong(expected, desired);
  delete job;
  inflight().fetch_sub(1);
  vTaskDelete(nullptr);
}

// Dispatch the quick-chat POST, tagged with the current quick_chat_generation.
inline bool quickchat_post_async(std::string url, std::string body, uint32_t timeout_ms, int generation) {
  qc_state().store(generation << 2);  // pending, tagged with this generation
  if (!spawn_worker<QcJob>("va_qc", qc_task,
                           new QcJob{std::move(url), std::move(body), timeout_ms, generation}, "quick_chat")) {
    qc_state().store((generation << 2) | 2);  // spawn dropped/failed -> surface as failure
    return false;
  }
  return true;
}

}  // namespace voice_dispatch
