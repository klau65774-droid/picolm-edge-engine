#include "threadpool.h"

#include <stdint.h>
#include <stdatomic.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
typedef HANDLE thread_t;
typedef unsigned (__stdcall *thread_entry)(void *);
#define THREAD_RESULT unsigned __stdcall
#define THREAD_RETURN 0
typedef SRWLOCK mutex_t;
typedef CONDITION_VARIABLE cond_t;
#define MUTEX_INIT SRWLOCK_INIT
#define COND_INIT CONDITION_VARIABLE_INIT
#define lock(m) AcquireSRWLockExclusive(m)
#define unlock(m) ReleaseSRWLockExclusive(m)
#define wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, 0)
#define wake_one(c) WakeConditionVariable(c)
#define wake_all(c) WakeAllConditionVariable(c)

static int thread_start(thread_t *thread, thread_entry entry, void *arg) {
    *thread = (HANDLE)_beginthreadex(NULL, 0, entry, arg, 0, NULL);
    return *thread != NULL;
}

static void thread_join(thread_t thread) {
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}
#else
#include <pthread.h>
typedef pthread_t thread_t;
typedef void *(*thread_entry)(void *);
#define THREAD_RESULT void *
#define THREAD_RETURN NULL
typedef pthread_mutex_t mutex_t;
typedef pthread_cond_t cond_t;
#define MUTEX_INIT PTHREAD_MUTEX_INITIALIZER
#define COND_INIT PTHREAD_COND_INITIALIZER
#define lock(m) pthread_mutex_lock(m)
#define unlock(m) pthread_mutex_unlock(m)
#define wait(c, m) pthread_cond_wait(c, m)
#define wake_one(c) pthread_cond_signal(c)
#define wake_all(c) pthread_cond_broadcast(c)

static int thread_start(thread_t *thread, thread_entry entry, void *arg) {
    return pthread_create(thread, NULL, entry, arg) == 0;
}

static void thread_join(thread_t thread) {
    pthread_join(thread, NULL);
}
#endif

typedef struct {
    threadpool_task task;
    void *context;
    int count;
    int slices;
} work_t;

static struct {
    mutex_t dispatch, mutex;
    cond_t available, done;
    thread_t workers[MAX_THREADS - 1];
    int ids[MAX_THREADS - 1];
    int worker_count;
    int persistent;
    int stopping;
    atomic_int threads;
    _Alignas(64) work_t work;
    uint64_t generation;
    _Alignas(64) int pending;
} pool = {
    .dispatch = MUTEX_INIT, .mutex = MUTEX_INIT,
    .available = COND_INIT, .done = COND_INIT, .threads = 1,
};

static void run_slice(const work_t *work, int slice) {
    int per_slice = work->count / work->slices;
    int extra = work->count % work->slices;
    int first = slice * per_slice + (slice < extra ? slice : extra);
    work->task(work->context, first, first + per_slice + (slice < extra));
}

static THREAD_RESULT persistent_worker(void *arg) {
    int slice = *(const int *)arg;
    uint64_t seen = 0;

    lock(&pool.mutex);
    for (;;) {
        while (!pool.stopping && seen == pool.generation) {
            wait(&pool.available, &pool.mutex);
        }
        if (pool.stopping) break;
        seen = pool.generation;
        if (slice >= pool.work.slices) continue;
        work_t work = pool.work;
        unlock(&pool.mutex);

        run_slice(&work, slice);

        lock(&pool.mutex);
        if (--pool.pending == 0) wake_one(&pool.done);
    }
    unlock(&pool.mutex);
    return THREAD_RETURN;
}

static void stop_workers(void) {
    if (pool.worker_count == 0) return;
    lock(&pool.mutex);
    pool.stopping = 1;
    wake_all(&pool.available);
    unlock(&pool.mutex);

    for (int i = 0; i < pool.worker_count; i++) thread_join(pool.workers[i]);
    pool.worker_count = 0;
    pool.stopping = 0;
    pool.generation = 0;
}

void threadpool_configure(int threads, int persistent) {
    if (threads < 1) threads = 1;
    if (threads > MAX_THREADS) threads = MAX_THREADS;
    lock(&pool.dispatch);
    stop_workers();
    pool.persistent = persistent != 0;
    if (pool.persistent) {
        for (int i = 0; i < threads - 1; i++) {
            pool.ids[i] = i + 1;
            if (!thread_start(&pool.workers[i], persistent_worker, &pool.ids[i])) break;
            pool.worker_count++;
        }
        threads = pool.worker_count + 1;
    }
    atomic_store_explicit(&pool.threads, threads, memory_order_relaxed);
    unlock(&pool.dispatch);
}

int threadpool_size(void) {
    return atomic_load_explicit(&pool.threads, memory_order_relaxed);
}

void threadpool_shutdown(void) {
    threadpool_configure(1, 0);
}

typedef struct {
    const work_t *work;
    int slice;
} transient_arg_t;

static THREAD_RESULT transient_worker(void *arg) {
    const transient_arg_t *worker = arg;
    run_slice(worker->work, worker->slice);
    return THREAD_RETURN;
}

static void run_transient(const work_t *work) {
    transient_arg_t args[MAX_THREADS];
    thread_t threads[MAX_THREADS];
    int started[MAX_THREADS];
    for (int i = 1; i < work->slices; i++) {
        args[i] = (transient_arg_t){work, i};
        started[i] = thread_start(&threads[i], transient_worker, &args[i]);
        if (!started[i]) run_slice(work, i);
    }
    run_slice(work, 0);
    for (int i = 1; i < work->slices; i++) {
        if (started[i]) thread_join(threads[i]);
    }
}

void threadpool_run(int count, threadpool_task task, void *context) {
    if (count <= 0) return;
    if (count == 1 || threadpool_size() == 1) {
        task(context, 0, count);
        return;
    }

    lock(&pool.dispatch);
    int threads = threadpool_size();
    work_t work = {task, context, count, count < threads ? count : threads};
    if (work.slices == 1) {
        run_slice(&work, 0);
    } else if (!pool.persistent) {
        run_transient(&work);
    } else {
        lock(&pool.mutex);
        pool.work = work;
        pool.pending = work.slices - 1;
        pool.generation++;
        wake_all(&pool.available);
        unlock(&pool.mutex);

        run_slice(&work, 0);

        lock(&pool.mutex);
        while (pool.pending) wait(&pool.done, &pool.mutex);
        unlock(&pool.mutex);
    }
    unlock(&pool.dispatch);
}
