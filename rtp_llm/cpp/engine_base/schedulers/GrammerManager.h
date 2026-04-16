#pragma once

#include <list>
#include <vector>
#include <string>
#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace py = pybind11;

namespace rtp_llm {

class GrammarManager {
public:
    explicit GrammarManager(py::object grammar_backend);
    ~GrammarManager();

    size_t size() const { return grammar_queue_.size(); }
    void   clear();
    bool   has_waiting_grammars() const { return !grammar_queue_.empty(); }

    bool process_req_with_grammar(const GenerateStreamPtr& stream);
    std::list<GenerateStreamPtr> get_ready_grammar_requests();
    void abort_requests(const GenerateStreamPtr& stream);
    void cleanupStream(const GenerateStreamPtr& stream);

private:
    bool isGrammarRequested(const GenerateStreamPtr& stream) const;
    py::tuple extractGrammarKey(const GenerateStreamPtr& stream) const;
    bool isInvalidGrammar(const py::object& obj) const;

    py::object grammar_backend_;
    py::object invalid_grammar_cls_ = py::none();
    double     grammar_poll_interval_s_     = 0.005;
    int32_t    grammar_max_poll_iterations_ = 10000;
    std::vector<GenerateStreamPtr> grammar_queue_;
};

}  // namespace rtp_llm
