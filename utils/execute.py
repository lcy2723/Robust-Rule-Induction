import re
import multiprocessing
import queue

class PythonExecutor:
    def __init__(self):
        self.result = None

    def execute(self, code: str, input_data: str, output_data: str):
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._worker,
            args=(input_data, code, output_data, result_queue)
        )
        process.start()
        process.join(timeout=10)  
        if process.is_alive():
            process.terminate()  
            process.join()
            self.result = "timeout"
        else:
            try:
                status, res = result_queue.get_nowait()
                self.result = res
            except queue.Empty:
                self.result = "timeout"
        return self.result

    def _worker(self, input_data, code, output_data, result_queue):
        context = {}
        try:
            # 执行输入数据、代码和输出数据
            exec(input_data, {}, context)
            exec(code, {}, context)
            exec(output_data, {}, context)
            result = context.get('result', None)
            result_queue.put(('success', result))
        except Exception as e:
            result_queue.put(('error', str(e)))

    
def extract_function_names(function_string):
    matches = re.findall(r"def (\w+)\(", function_string)
    return matches

def extract_function(function_string):
    function_string = re.search(r"```python(.*?)```", function_string, re.DOTALL).group(1).strip()
    return function_string