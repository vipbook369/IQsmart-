# memory_handler.py
"""
IQsmartBot ke liye Memory Handler Module
Ye module PRNG inputs, predictions, feedback aur session data ko store/manage karega.
"""

import json
import os
from datetime import datetime

class MemoryHandler:
    def __init__(self, memory_file="iqsmart_memory.json"):
        self.memory_file = memory_file
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "sessions": [],
                "feedback": {},
                "predictions": [],
                "timestamp": str(datetime.now())
            }

    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def add_session(self, numbers):
        self.memory["sessions"].append(numbers)
        self.memory["timestamp"] = str(datetime.now())
        self.save_memory()

    def get_last_session(self):
        return self.memory["sessions"][-1] if self.memory["sessions"] else []

    def add_prediction(self, prediction):
        self.memory["predictions"].append({
            "timestamp": str(datetime.now()),
            "prediction": prediction
        })
        self.save_memory()

    def add_feedback(self, number, is_correct):
        if number not in self.memory["feedback"]:
            self.memory["feedback"][number] = {"correct": 0, "incorrect": 0}

        if is_correct:
            self.memory["feedback"][number]["correct"] += 1
        else:
            self.memory["feedback"][number]["incorrect"] += 1

        self.save_memory()

    def get_feedback(self, number):
        return self.memory["feedback"].get(number, {"correct": 0, "incorrect": 0})

    def get_all_sessions(self):
        return self.memory["sessions"]

    def clear_memory(self):
        self.memory = {
            "sessions": [],
            "feedback": {},
            "predictions": [],
            "timestamp": str(datetime.now())
        }
        self.save_memory()


# Example use:
if __name__ == "__main__":
    mh = MemoryHandler()
    mh.add_session([1, 2, 3, 4, 5])
    print("Last session:", mh.get_last_session())
    mh.add_prediction("Big & Green")
    mh.add_feedback(5, True)
    print("Feedback for 5:", mh.get_feedback(5))
