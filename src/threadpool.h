#ifndef THREADPOOL_H
#define THREADPOOL_H

#define MAX_THREADS 16

typedef void (*threadpool_task)(void *context, int first, int end);

/* The calling thread works on the first slice. Disabling persistence keeps
 * the configured thread count and uses temporary workers for comparison. */
void threadpool_configure(int threads, int persistent);
int threadpool_size(void);
void threadpool_run(int count, threadpool_task task, void *context);
void threadpool_shutdown(void);

#endif
