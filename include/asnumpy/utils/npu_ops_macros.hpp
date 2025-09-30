
#include <asnumpy/utils/status_handler.hpp>
#include <fmt/base.h>

#define CHECK_RET(cond, return_expr)                                                                                   \
	do {                                                                                                               \
		if (!(cond)) {                                                                                                 \
			return_expr;                                                                                               \
		}                                                                                                              \
	}                                                                                                                  \
	while (0)

#define LOG_PRINT(message, ...)                                                                                        \
	do {                                                                                                               \
		throw std::runtime_error(fmt::format(message, ##__VA_ARGS__));                                                 \
	}                                                                                                                  \
	while (0)


#define EXECUTE_OP_WORKSPACE(OpName, workspaceSize, executor, AclnnFunc)                                               \
	do {                                                                                                               \
                                                                                                                       \
		void* workspaceAddr = nullptr;                                                                                 \
		if (workspaceSize > 0) {                                                                                       \
			auto error_malloc = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);                 \
			CheckMallocAclnnStatus(error_malloc);                                                                      \
		}                                                                                                              \
		auto error_func = AclnnFunc(workspaceAddr, workspaceSize, executor, nullptr);                                  \
		CheckAclnnStatus(error_func, fmt::format("[{}] Failed to execute operation.", #OpName));                       \
		auto error_sync = aclrtSynchronizeDevice();                                                                    \
		CheckSynchronizeDeviceAclnnStatus(error_sync);                                                                 \
	}                                                                                                                  \
	while (0)

#define DEFINE_UNARY_OP(OpName, AclnnGetWorkspaceSizeFunc, AclnnFunc)                                                  \
	NPUArray OpName(const NPUArray& x) {                                                                               \
		auto shape = x.shape;                                                                                          \
		auto dtype = x.dtype;                                                                                          \
		auto result = NPUArray(shape, dtype);                                                                          \
		uint64_t workspaceSize = 0;                                                                                    \
		aclOpExecutor* executor;                                                                                       \
		auto error = AclnnGetWorkspaceSizeFunc(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);              \
		CheckGetWorkspaceSizeAclnnStatus(error);                                                                       \
		EXECUTE_OP_WORKSPACE(OpName, workspaceSize, executor, AclnnFunc);                                              \
		return result;                                                                                                 \
	}

#define DEFINE_BINARY_OP(OpName, AclnnGetWorkspaceSizeFunc, AclnnFunc)                                                 \
	NPUArray OpName(const NPUArray& x1, const NPUArray& x2) {                                                          \
		auto shape = GetBroadcastShape(x1, x2);                                                                        \
		auto dtype = x1.dtype;                                                                                         \
		auto result = NPUArray(shape, dtype);                                                                          \
		uint64_t workspaceSize = 0;                                                                                    \
		aclOpExecutor* executor;                                                                                       \
		auto error =                                                                                                   \
			AclnnGetWorkspaceSizeFunc(x1.tensorPtr, x2.tensorPtr, result.tensorPtr, &workspaceSize, &executor);        \
		CheckGetWorkspaceSizeAclnnStatus(error);                                                                       \
		EXECUTE_OP_WORKSPACE(OpName, workspaceSize, executor, AclnnFunc);                                              \
		return result;                                                                                                 \
	}
