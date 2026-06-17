import heapq
from typing import List, Dict


def compare_batching(
    requests: List[int],
    max_batch_size: int,
    time_per_step: float
) -> Dict[str, float]:
    
    if not requests:
        return {
            "static_total_time": 0.0,
            "continuous_total_time": 0.0,
            "static_throughput": 0.0,
            "continuous_throughput": 0.0,
            "static_gpu_utilization": 0.0,
            "continuous_gpu_utilization": 0.0,
            "speedup": 0.0,
        }

    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    
    if time_per_step <= 0:
        raise ValueError("time_per_step must be positive")
    
    total_requests = len(requests)
    total_useful_tokens = sum(requests)
    
    static_total_steps = 0
    
    for i in range(0, total_requests, max_batch_size):
        batch = requests[i:i + max_batch_size]
        static_total_steps += max(batch)
        
    static_total_time = static_total_steps * time_per_step
    
    slots = [0] * max_batch_size
    heapq.heapify(slots)
    
    for req_steps in requests:
        earliest_free_time = heapq.heappop(slots)
        finish_time = earliest_free_time + req_steps
        heapq.heappush(slots, finish_time)
        
    continuous_total_steps = max(slots)
    continuous_total_time = continuous_total_steps * time_per_step
    
    static_throughput = total_requests / (static_total_time / 1000)
    continuous_throughput = total_requests / (continuous_total_time / 1000)
    
    static_gpu_utilization = total_useful_tokens / (
        static_total_steps * max_batch_size
    )
    
    continuous_gpu_utilization = total_useful_tokens / (
        continuous_total_steps * max_batch_size
    )
    
    speedup = continuous_throughput / static_throughput
    
    return {
        "static_total_time": round(static_total_time, 4),
        "continuous_total_time": round(continuous_total_time, 4),
        "static_throughput": round(static_throughput, 4),
        "continuous_throughput": round(continuous_throughput, 4),
        "static_gpu_utilization": round(static_gpu_utilization, 4),
        "continuous_gpu_utilization": round(continuous_gpu_utilization, 4),
        "speedup": round(speedup, 4),
    }


if __name__ == "__main__":
    requests = [10, 2, 8, 3, 5, 1, 7, 4]
    max_batch_size = 4
    time_per_step = 50.0  # ms

    print(f"Requests (steps each): {requests}")
    print(f"Max batch size: {max_batch_size}")
    print(f"Time per step: {time_per_step} ms\n")

    results = compare_batching(requests, max_batch_size, time_per_step)
    for k, v in results.items():
        print(f"  {k}: {v}")
