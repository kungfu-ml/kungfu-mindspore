#include "backend/kernel_compiler/cpu/kungfu/kungfu_logger.h"

namespace kungfu
{
std::mutex logger::mu_;

bool logger::show_thread_id = true;

std::set<log_once::ukey> log_once::sites_;

std::map<std::pair<std::string, int>, int> log_every::sites_;

int scope_logger::depth = 0;
}  // namespace kungfu
