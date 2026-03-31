load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "cp_cache_scatter_kernel_test",
    srcs = [
        "CPCacheScatterKernelTest.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/kernels:cp_cache_scatter",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    exec_properties = {'gpu': 'A10'},
)

cc_test(
    name = "virtual_cache_key_test",
    srcs = [
        "VirtualCacheKeyTest.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/utils:hash_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
