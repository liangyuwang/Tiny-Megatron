# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import time

class RuntimeAutoTuner:
    def __init__(self, enable=True, warmup_iterations=10, measure_iterations=100, verbose=False) -> None:
        self.enable = enable
        self.if_final_tune = False
        self.chosen_funcs = {}  # Dictionary to store chosen functions for different function lists
        self.warmup_iterations = warmup_iterations
        self.measure_iterations = measure_iterations
        self.verbose = verbose

    def choose_function(self, funcs_list, *args, **kwargs):
        if not self.enable:
            return funcs_list[0]
        
        # Create a key based on the function list to identify different function types
        func_key = tuple(func.__name__ for func in funcs_list)
        
        if func_key in self.chosen_funcs and self.if_final_tune:
            return self.chosen_funcs[func_key]
        
        time_list = []
        for func in funcs_list:
            for _ in range(self.warmup_iterations):
                func(*args, **kwargs)
            time_list.append(self._measure_time(func, *args, **kwargs))
        
        chosen_func = funcs_list[time_list.index(min(time_list))]
        self.chosen_funcs[func_key] = chosen_func
        
        if self.verbose:
            print(f"Chosen function for {func_key}: {chosen_func}")
        return chosen_func

    def final_tune(self):
        self.if_final_tune = True

    def _measure_time(self, func, *args, **kwargs):
        start = time.time()
        for _ in range(self.measure_iterations):
            func(*args, **kwargs)
        end = time.time()
        return (end - start) / self.measure_iterations
