#include "rtp_llm/cpp/engine_base/schedulers/GrammerManager.h"
#include <chrono>
#include <cstdlib>
#include <set>
#include <string>
#include <thread>
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {
namespace {

std::string pyObjTypeName(const py::object& obj) {
    try {
        if (obj.is_none()) {
            return "None";
        }
        py::object cls    = obj.attr("__class__");
        std::string mod   = py::str(cls.attr("__module__")).cast<std::string>();
        std::string cname = py::str(cls.attr("__name__")).cast<std::string>();
        return mod + "." + cname;
    } catch (...) {
        return "<unknown_py_type>";
    }
}

std::string grammarKeyBrief(const py::tuple& key) {
    try {
        if (key.size() < 2) {
            return "<empty_key>";
        }
        std::string key_type = py::str(key[0]).cast<std::string>();
        std::string key_text = py::str(key[1]).cast<std::string>();
        return key_type + "(len=" + std::to_string(key_text.size()) + ")";
    } catch (...) {
        return "<invalid_key>";
    }
}

bool isLikelyXGrammarObject(const py::object& obj) {
    std::string type_name = pyObjTypeName(obj);
    return type_name.find("xgram") != std::string::npos || type_name.find("XGram") != std::string::npos
           || type_name.find("XGrammar") != std::string::npos;
}

}  // namespace

GrammarManager::GrammarManager(py::object grammar_backend)
    : grammar_backend_(std::move(grammar_backend)) {
    py::gil_scoped_acquire acquire;
    if (!grammar_backend_.is_none() && py::hasattr(grammar_backend_, "_invalid_grammar_cls")) {
        invalid_grammar_cls_ = grammar_backend_.attr("_invalid_grammar_cls");
    }
    if (const char* v = std::getenv("SGLANG_GRAMMAR_POLL_INTERVAL")) {
        try { grammar_poll_interval_s_ = std::stod(v); } catch (...) {}
    }
    if (const char* v = std::getenv("SGLANG_GRAMMAR_MAX_POLL_ITERATIONS")) {
        try { grammar_max_poll_iterations_ = std::stoi(v); } catch (...) {}
    }
    RTP_LLM_LOG_INFO("GrammarManager init: backend_type=%s, poll_interval=%.4fs, max_poll_iterations=%d",
                     pyObjTypeName(grammar_backend_).c_str(),
                     grammar_poll_interval_s_,
                     grammar_max_poll_iterations_);
}

GrammarManager::~GrammarManager() {
    clear();
    py::gil_scoped_acquire acquire;
    invalid_grammar_cls_ = py::none();
    grammar_backend_     = py::none();
}

void GrammarManager::clear() {
    py::gil_scoped_acquire acquire;
    RTP_LLM_LOG_INFO("GrammarManager clear: queue_size_before=%zu", grammar_queue_.size());
    if (!grammar_backend_.is_none()) {
        try {
            grammar_backend_.attr("reset")();
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_WARNING("grammar backend reset failed: %s", e.what());
        }
    }
    for (auto& stream : grammar_queue_) {
        if (stream && stream->hasGrammarFuture()) {
            try {
                stream->grammarFuture().attr("cancel")();
            } catch (...) {
            }
        }
        if (stream) {
            stream->clearGrammarFuture();
            stream->clearGrammarKey();
            stream->resetGrammarWaitCount();
            stream->clearGrammarObject();
        }
    }
    grammar_queue_.clear();
}

bool GrammarManager::isGrammarRequested(const GenerateStreamPtr& stream) const {
    auto& config = stream->generateConfig();
    return config->json_schema.has_value() || config->regex.has_value() || config->ebnf.has_value()
           || config->structural_tag.has_value();
}

py::tuple GrammarManager::extractGrammarKey(const GenerateStreamPtr& stream) const {
    auto& config = stream->generateConfig();
    if (config->json_schema.has_value()) {
        return py::make_tuple("json", config->json_schema.value());
    } else if (config->regex.has_value()) {
        return py::make_tuple("regex", config->regex.value());
    } else if (config->ebnf.has_value()) {
        return py::make_tuple("ebnf", config->ebnf.value());
    } else if (config->structural_tag.has_value()) {
        return py::make_tuple("structural_tag", config->structural_tag.value());
    }
    return py::tuple();
}

bool GrammarManager::isInvalidGrammar(const py::object& obj) const {
    if (invalid_grammar_cls_.is_none()) {
        return false;
    }
    return py::isinstance(obj, invalid_grammar_cls_);
}

bool GrammarManager::process_req_with_grammar(const GenerateStreamPtr& stream) {
    py::gil_scoped_acquire acquire;

    if (!isGrammarRequested(stream)) {
        stream->clearGrammarObject();
        RTP_LLM_LOG_DEBUG("stream [%ld] no grammar constraints, bypass grammar queue", stream->streamId());
        return false;
    }

    if (grammar_backend_.is_none()) {
        stream->setStop(ErrorCode::INVALID_PARAMS,
                        "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not "
                        "supported when the server is launched with --grammar-backend none");
        return false;
    }

    auto key = extractGrammarKey(stream);
    RTP_LLM_LOG_DEBUG("stream [%ld] grammar preprocess begin: key=%s, require_reasoning=%d",
                      stream->streamId(),
                      grammarKeyBrief(key).c_str(),
                      static_cast<int>(stream->generateConfig()->in_think_mode));
    try {
        bool require_reasoning = stream->generateConfig()->in_think_mode;
        py::object result      = grammar_backend_.attr("get_cached_or_future_value")(key, require_reasoning);
        py::object value       = result.cast<py::tuple>()[0];
        bool       cache_hit   = result.cast<py::tuple>()[1].cast<bool>();
        RTP_LLM_LOG_DEBUG("stream [%ld] grammar backend returned: cache_hit=%d, value_type=%s",
                          stream->streamId(),
                          static_cast<int>(cache_hit),
                          pyObjTypeName(value).c_str());

        if (cache_hit) {
            if (isInvalidGrammar(value)) {
                std::string err = py::str(value.attr("error_message")).cast<std::string>();
                std::string key_type = py::str(key[0]).cast<std::string>();
                stream->setStop(ErrorCode::INVALID_PARAMS, "Failed to compile " + key_type + " grammar: " + err);
                return false;
            }
            stream->setGrammarObject(value);
            stream->clearGrammarFuture();
            stream->clearGrammarKey();
            stream->resetGrammarWaitCount();
            RTP_LLM_LOG_INFO("stream [%ld] grammar cache hit accepted: key=%s, grammar_type=%s, likely_xgrammar=%d",
                             stream->streamId(),
                             grammarKeyBrief(key).c_str(),
                             pyObjTypeName(value).c_str(),
                             static_cast<int>(isLikelyXGrammarObject(value)));
            return false;
        }

        stream->setGrammarFuture(value);
        stream->setGrammarKey(key);
        stream->resetGrammarWaitCount();
        grammar_queue_.emplace_back(stream);
        RTP_LLM_LOG_INFO("stream [%ld] grammar async compile queued: key=%s, future_type=%s, queue_size=%zu",
                         stream->streamId(),
                         grammarKeyBrief(key).c_str(),
                         pyObjTypeName(value).c_str(),
                         grammar_queue_.size());
        return true;
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_WARNING("stream [%ld] grammar backend exception: %s", stream->streamId(), e.what());
        stream->setStop(ErrorCode::INVALID_PARAMS, std::string("grammar backend error: ") + e.what());
        return false;
    }
}

std::list<GenerateStreamPtr> GrammarManager::get_ready_grammar_requests() {
    std::list<GenerateStreamPtr> return_reqs;
    if (grammar_queue_.empty()) {
        return return_reqs;
    }
    RTP_LLM_LOG_INFO("grammar poll begin: queue_size=%zu", grammar_queue_.size());

    std::set<size_t> ready_idxs;
    std::set<size_t> failed_idxs;

    auto start     = std::chrono::steady_clock::now();
    auto sleep_dur = std::chrono::duration<double>(grammar_poll_interval_s_ / 10.0);

    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
           < grammar_poll_interval_s_) {
        for (size_t i = 0; i < grammar_queue_.size(); ++i) {
            if (ready_idxs.count(i)) {
                continue;
            }
            auto& stream = grammar_queue_[i];

            if (!stream || stream->finished() || stream->stopped() || !stream->hasGrammarFuture()) {
                ready_idxs.insert(i);
                continue;
            }

            try {
                py::gil_scoped_acquire acquire;
                bool done = stream->grammarFuture().attr("done")().cast<bool>();
                if (done) {
                    ready_idxs.insert(i);
                    RTP_LLM_LOG_INFO("stream [%ld] grammar future done in poll window", stream->streamId());
                } else if (stream->grammarWaitCount() == 0 || stream->grammarWaitCount() % 50 == 0) {
                    RTP_LLM_LOG_DEBUG("stream [%ld] grammar future pending: wait_count=%d",
                                      stream->streamId(),
                                      stream->grammarWaitCount());
                }
            } catch (const py::error_already_set& e) {
                RTP_LLM_LOG_WARNING("stream [%ld] grammar done check exception: %s",
                                    stream->streamId(),
                                    e.what());
                stream->setStop(ErrorCode::INVALID_PARAMS, std::string("grammar compile error: ") + e.what());
                ready_idxs.insert(i);
            }
        }
        if (ready_idxs.size() == grammar_queue_.size()) {
            break;
        }
        std::this_thread::sleep_for(sleep_dur);
    }

    for (size_t i = 0; i < grammar_queue_.size(); ++i) {
        if (ready_idxs.count(i)) {
            continue;
        }
        auto& stream = grammar_queue_[i];
        if (!stream) {
            failed_idxs.insert(i);
            continue;
        }
        stream->incGrammarWaitCount();
        if (stream->grammarWaitCount() >= grammar_max_poll_iterations_) {
            failed_idxs.insert(i);
            RTP_LLM_LOG_WARNING("stream [%ld] grammar wait timeout: wait_count=%d, max=%d",
                                stream->streamId(),
                                stream->grammarWaitCount(),
                                grammar_max_poll_iterations_);
        }
    }

    for (auto i : ready_idxs) {
        auto& stream = grammar_queue_[i];
        if (!stream) {
            continue;
        }
        return_reqs.emplace_back(stream);

        if (stream->finished() || stream->stopped() || !stream->hasGrammarFuture()) {
            stream->clearGrammarFuture();
            stream->clearGrammarKey();
            stream->resetGrammarWaitCount();
            continue;
        }
        try {
            py::gil_scoped_acquire acquire;
            py::object grammar_obj = stream->grammarFuture().attr("result")();
            py::tuple  grammar_key = stream->grammarKey();
            RTP_LLM_LOG_INFO("stream [%ld] grammar future result: key=%s, grammar_type=%s, likely_xgrammar=%d",
                             stream->streamId(),
                             grammarKeyBrief(grammar_key).c_str(),
                             pyObjTypeName(grammar_obj).c_str(),
                             static_cast<int>(isLikelyXGrammarObject(grammar_obj)));
            if (isInvalidGrammar(grammar_obj)) {
                std::string key_type = py::str(grammar_key[0]).cast<std::string>();
                std::string err      = py::str(grammar_obj.attr("error_message")).cast<std::string>();
                grammar_backend_.attr("set_cache")(grammar_key, grammar_obj);
                stream->setStop(ErrorCode::INVALID_PARAMS, "Failed to compile " + key_type + " grammar: " + err);
                stream->clearGrammarFuture();
                stream->clearGrammarKey();
                stream->resetGrammarWaitCount();
                continue;
            }
            grammar_backend_.attr("set_cache")(grammar_key, grammar_obj.attr("copy")());
            stream->setGrammarObject(grammar_obj);
            RTP_LLM_LOG_INFO("stream [%ld] grammar ready -> waiting candidate: key=%s", stream->streamId(), grammarKeyBrief(grammar_key).c_str());
            stream->clearGrammarFuture();
            stream->clearGrammarKey();
            stream->resetGrammarWaitCount();
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_WARNING("stream [%ld] grammar future exception: %s", stream->streamId(), e.what());
            stream->setStop(ErrorCode::INVALID_PARAMS, std::string("grammar compile error: ") + e.what());
            stream->clearGrammarFuture();
            stream->clearGrammarKey();
            stream->resetGrammarWaitCount();
        }
    }

    for (auto i : failed_idxs) {
        auto& stream = grammar_queue_[i];
        if (!stream) {
            continue;
        }
        return_reqs.emplace_back(stream);
        try {
            py::gil_scoped_acquire acquire;
            if (stream->hasGrammarFuture()) {
                stream->grammarFuture().attr("cancel")();
            }
        } catch (...) {
        }

        if (!invalid_grammar_cls_.is_none()) {
            try {
                py::gil_scoped_acquire acquire;
                py::object invalid = invalid_grammar_cls_("Grammar preprocessing timed out");
                if (stream->hasGrammarKey()) {
                    grammar_backend_.attr("set_cache")(stream->grammarKey(), invalid);
                }
            } catch (const py::error_already_set& e) {
                RTP_LLM_LOG_WARNING("set timeout invalid grammar cache failed: %s", e.what());
            }
        }
        stream->setStop(ErrorCode::GENERATE_TIMEOUT, "Grammar preprocessing timed out");
        stream->clearGrammarFuture();
        stream->clearGrammarKey();
        stream->resetGrammarWaitCount();
    }

    if (!ready_idxs.empty() || !failed_idxs.empty()) {
        std::vector<GenerateStreamPtr> new_queue;
        new_queue.reserve(grammar_queue_.size());
        for (size_t i = 0; i < grammar_queue_.size(); ++i) {
            if (ready_idxs.count(i) == 0 && failed_idxs.count(i) == 0) {
                new_queue.emplace_back(std::move(grammar_queue_[i]));
            }
        }
        grammar_queue_ = std::move(new_queue);
    }
    RTP_LLM_LOG_INFO("grammar poll end: ready=%zu, failed=%zu, queue_size_after=%zu",
                     ready_idxs.size(),
                     failed_idxs.size(),
                     grammar_queue_.size());
    return return_reqs;
}

void GrammarManager::abort_requests(const GenerateStreamPtr& stream) {
    if (!stream) {
        return;
    }
    for (auto& queued : grammar_queue_) {
        if (queued == stream) {
            RTP_LLM_LOG_DEBUG("abort grammar queue request, stream [%ld]", stream->streamId());
            py::gil_scoped_acquire acquire;
            if (stream->hasGrammarFuture()) {
                try {
                    stream->grammarFuture().attr("cancel")();
                } catch (...) {
                }
            }
            stream->setStop(ErrorCode::CANCELLED, "Aborted");
            break;
        }
    }
}

void GrammarManager::cleanupStream(const GenerateStreamPtr& stream) {
    if (!stream) {
        return;
    }
    py::gil_scoped_acquire acquire;
    try {
        if (stream->hasGrammarFuture()) {
            stream->grammarFuture().attr("cancel")();
        }
    } catch (...) {
    }
    stream->clearGrammarObject();
    stream->clearGrammarFuture();
    stream->clearGrammarKey();
    stream->resetGrammarWaitCount();
    for (auto it = grammar_queue_.begin(); it != grammar_queue_.end(); ++it) {
        if (*it == stream) {
            grammar_queue_.erase(it);
            return;
        }
    }
}

}  // namespace rtp_llm
