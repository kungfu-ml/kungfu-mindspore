#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <thread>

namespace kungfu
{
class log_once;
class log_every;

class logger
{
    static std::mutex mu_;

    friend class log_once;
    friend class log_every;

    std::lock_guard<std::mutex> lk_;
    std::ostream &os_;

  public:
    static bool show_thread_id;

    logger(std::ostream &os = std::cout) : lk_(mu_), os_(os)
    {
        os_ << "[D]";
        if (show_thread_id) {
            (*this) << std::this_thread::get_id();
        }
    }

    ~logger()
    {
        os_ << std::endl;
    }

    template <typename T>
    logger &operator<<(const T &x)
    {
        os_ << ' ' << x;
        return *this;
    }
};

class log_once
{
    using ukey = std::tuple<int, std::string, std::string>;
    static std::set<ukey> sites_;

    std::lock_guard<std::mutex> lk_;
    std::ostream &os_;
    bool first_;

  public:
    log_once(std::string filename, int line, std::string key,
             std::ostream &os = std::cout)
        : lk_(logger::mu_),
          os_(os),
          first_(sites_.count(ukey(line, key, filename)) == 0)
    {
        if (first_) {
            sites_.insert(ukey(line, key, filename));
            os_ << "[D]";
            if (logger::show_thread_id) {
                (*this) << std::this_thread::get_id();
            }
        }
    }

    ~log_once()
    {
        if (first_) {
            os_ << std::endl;
        }
    }

    template <typename T>
    log_once &operator<<(const T &x)
    {
        if (first_) {
            os_ << ' ' << x;
        }
        return *this;
    }
};

class log_every
{
    static std::map<std::pair<std::string, int>, int> sites_;

    const int period_;
    std::lock_guard<std::mutex> lk_;
    std::ostream &os_;
    int count_;

  public:
    log_every(std::string filename, int line, int period,
              std::ostream &os = std::cout)
        : period_(period),
          lk_(logger::mu_),
          os_(os),
          count_(sites_[std::make_pair(filename, line)])
    {
        ++sites_[std::make_pair(filename, line)];
        if (count_ % period_ == 0) {
            os_ << "[D]";
            if (logger::show_thread_id) {
                (*this) << std::this_thread::get_id();
            }
        }
    }

    ~log_every()
    {
        if (count_ % period_ == 0) {
            os_ << std::endl;
        }
    }

    template <typename T>
    log_every &operator<<(const T &x)
    {
        if (count_ % period_ == 0) {
            os_ << ' ' << x;
        }
        return *this;
    }
};

class scope_logger
{
    std::string name_;
    std::string filename_;
    int line_;
    std::ostream &os_;

    static int depth;

  public:
    scope_logger(std::string name, std::string filename, int line,
                 std::ostream &os = std::cout)
        : name_(std::move(name)),
          filename_(std::move(filename)),
          line_(line),
          os_(os)
    {
        os_ << std::string(depth * 4, ' ') << "{ // " << name_ << ' '
            << filename_ << ':' << line_ << std::endl;
        ++depth;
    }

    ~scope_logger()
    {
        --depth;
        os_ << std::string(depth * 4, ' ') << "} // " << name_ << ' '
            << filename_ << ':' << line_ << std::endl;
    }
};

class scope_duration_logger
{
    using Clock = std::chrono::high_resolution_clock;
    using duration_t = std::chrono::duration<double>;
    using instant_t = std::chrono::time_point<Clock>;

    std::string name_;
    std::string filename_;
    int line_;
    std::ostream &os_;
    const instant_t t0_;

  public:
    scope_duration_logger(std::string name, std::string filename, int line,
                          std::ostream &os = std::cout)
        : name_(std::move(name)),
          filename_(std::move(filename)),
          line_(line),
          os_(os),
          t0_(Clock::now())
    {
    }

    ~scope_duration_logger()
    {
        duration_t d = Clock::now() - t0_;
        os_ << " took: " << std::setw(8) << d.count() * 1e3 << "ms"
            << "        @ " << name_;
        // os_ << " ! " << filename_ << ':' << line_;
        os_ << std::endl;
    }
};
}  // namespace kungfu

#define KF_LOG(...) kungfu::logger(__VA_ARGS__)

#define KF_LOG_ONCE(key) kungfu::log_once(__FILE__, __LINE__, key)

#define KF_LOG_EVERY(n) kungfu::log_every(__FILE__, __LINE__, n)

#define KF_LOG_SCOPE_LOC(name)                                                 \
    kungfu::scope_logger _scope_logger(name, __FILE__, __LINE__);

#define KF_LOG_SCOPE_DURATION(name)                                            \
    kungfu::scope_duration_logger _scope_duration_logger(name, __FILE__,       \
                                                         __LINE__);
