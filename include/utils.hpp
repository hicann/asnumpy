/******************************************************************************
 * Copyright [2024]-[2025] [HIT1920/asnumpy] Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#pragma once

#include <cstdint>
#include <acl/acl.h>
#include <stdexcept>

class Workspace {
private:
    uint64_t workspaceSize_;
    void* workspaceAddr_;

    // 禁止拷贝
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

public:
    // 显式构造函数进行资源分配
    explicit Workspace(uint64_t size = 0) 
        : workspaceSize_(size), workspaceAddr_(nullptr) {
        if (workspaceSize_ > 0) {
            auto error = aclrtMalloc(&workspaceAddr_, workspaceSize_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (error != ACL_SUCCESS || !workspaceAddr_) {
                // 分配失败时重置状态
                workspaceSize_ = 0;
                workspaceAddr_ = nullptr;
            }
        }
    }

    // 移动构造函数
    Workspace(Workspace&& other) noexcept
        : workspaceSize_(other.workspaceSize_), workspaceAddr_(other.workspaceAddr_) {
        other.workspaceSize_ = 0;
        other.workspaceAddr_ = nullptr;
    }

    // 移动赋值运算符
    Workspace& operator=(Workspace&& other) noexcept {
        if (this != &other) {
            // 释放当前资源
            if (workspaceAddr_ && workspaceSize_ > 0) {
                aclrtFree(workspaceAddr_);
            }
            // 转移资源
            workspaceSize_ = other.workspaceSize_;
            workspaceAddr_ = other.workspaceAddr_;
            other.workspaceSize_ = 0;
            other.workspaceAddr_ = nullptr;
        }
        return *this;
    }

    // 析构函数自动释放资源
    ~Workspace() {
        if (workspaceAddr_ && workspaceSize_ > 0) {
            aclrtFree(workspaceAddr_);
        }
    }

    // 获取方法保持const正确性
    uint64_t GetWorkspaceSize() const {
        return workspaceSize_;
    }

    // 返回非常量指针以满足ACL接口要求
    void* GetWorkspaceAddr() {
        return workspaceAddr_;
    }

    // 保留const版本用于只读场景
    const void* GetConstWorkspaceAddr() const {
        return workspaceAddr_;
    }

    // 重置资源（可用于动态改变大小）
    void Reset(uint64_t newSize = 0) {
        if (workspaceAddr_ && workspaceSize_ > 0) {
            aclrtFree(workspaceAddr_);
        }
        workspaceSize_ = newSize;
        workspaceAddr_ = nullptr;
        
        if (workspaceSize_ > 0) {
            auto error = aclrtMalloc(&workspaceAddr_, workspaceSize_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (error != ACL_SUCCESS || !workspaceAddr_) {
                // 分配失败时重置状态
                workspaceSize_ = 0;
                workspaceAddr_ = nullptr;
            }
        }
    }
};
