
from io import StringIO
from contextlib import redirect_stdout
import multiprocessing
from multiprocessing import Pool
import threading
# multiprocessing.set_start_method('fork')


def format_code(code_str: str):
    code = "def run_it():\n"
    for line in code_str.split("\n"):
        code += "  " + line + "\n"
    code += "run_it()"
    return code

class CodeExecutor:

    def __init__(self, code, timeout, use_process: bool):
        self.code = format_code(code)
        self.timeout = timeout
        self.error = ""
        self.use_process = use_process

    def execute_code(self, return_val):
        try:
            # print("111")
            f = StringIO()
            # import pdb; pdb.set_trace()
            with redirect_stdout(f):
                exec(self.code, globals(), locals())
            s = f.getvalue()
            s = s.strip("\n")
            return_val["result"] = s
        except Exception as e:
            self.error = e
            return_val["error"] = e
            # print(e)
            pass

    @staticmethod
    def execute_code_with_string(code, index, return_val):
        code = format_code(code)
        try:
            f = StringIO()
            with redirect_stdout(f):
                exec(code, globals(), locals())
            s = f.getvalue()
            s = s.strip("\n")
            return_val[index] = s
        except Exception as e:
            # print(e)
            pass

    def run(self):
        if self.use_process:
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            process = multiprocessing.Process(target=self.execute_code,
                                              args=(return_dict, ))
            process.start()
            process.join(timeout=self.timeout)
            process.terminate()
        else:
            return_dict = {}
            thread = threading.Thread(target=self.execute_code,
                                      args=(return_dict, ))
            thread.start()
            thread.join(timeout=self.timeout)
            if thread.is_alive():
                print("time out!")
                
                self.error = "Execution timed out"
                return return_dict

        return return_dict
        
