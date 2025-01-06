import sys
import time
import psutil
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class DependencyGraph:
    def __init__(self):
        self.adj_list = {}
        self.tasks = set()
        self.graph_nx = nx.DiGraph()

    def add_task(self, task_name):
        if task_name in self.tasks:
            raise ValueError(f"Task '{task_name}' already exists.")
        self.tasks.add(task_name)
        self.adj_list[task_name] = []
        self.graph_nx.add_node(task_name)

    def add_dependency(self, from_task, to_task):
        if from_task not in self.tasks:
            raise ValueError(f"Task '{from_task}' does not exist.")
        if to_task not in self.tasks:
            raise ValueError(f"Task '{to_task}' does not exist.")

        self.adj_list[from_task].append(to_task)
        self.graph_nx.add_edge(from_task, to_task)

    def detect_cycle_dfs(self):
        visited = set()
        recursion_stack = set()

        def dfs(task):
            visited.add(task)
            recursion_stack.add(task)
            for neighbor in self.adj_list[task]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True
            recursion_stack.remove(task)
            return False

        for task in self.tasks:
            if task not in visited:
                if dfs(task):
                    return True
        return False

    def topological_sort(self):
        visited = set()
        stack = []

        def dfs_topo(task):
            visited.add(task)
            for neighbor in self.adj_list[task]:
                if neighbor not in visited:
                    dfs_topo(neighbor)
            stack.append(task)

        for task in self.tasks:
            if task not in visited:
                dfs_topo(task)

        return stack[::-1]

    def draw_graph(self, title="Task Dependency Graph"):
        pos = nx.spring_layout(self.graph_nx)
        nx.draw_networkx(self.graph_nx, pos, with_labels=True, node_color='lightblue', 
                         arrows=True, arrowstyle='->', arrowsize=12)
        plt.title(title)
        plt.show()

class TwoStacksOneArray:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.array = [None] * capacity
        self.top1 = -1
        self.top2 = capacity

    def push1(self, value):
        if self.top1 < self.top2 - 1:
            self.top1 += 1
            self.array[self.top1] = value
        else:
            raise OverflowError("Stack1 Overflow")

    def push2(self, value):
        if self.top1 < self.top2 - 1:
            self.top2 -= 1
            self.array[self.top2] = value
        else:
            raise OverflowError("Stack2 Overflow")

    def pop1(self):
        if self.top1 >= 0:
            val = self.array[self.top1]
            self.array[self.top1] = None
            self.top1 -= 1
            return val
        else:
            raise IndexError("Pop from empty Stack1")

    def pop2(self):
        if self.top2 < self.capacity:
            val = self.array[self.top2]
            self.array[self.top2] = None
            self.top2 += 1
            return val
        else:
            raise IndexError("Pop from empty Stack2")

    def __repr__(self):
        return f"<TwoStacksOneArray top1={self.top1}, top2={self.top2}, array={self.array}>"


class QueueArray:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.array = [None] * capacity
        self.front = 0
        self.rear = -1
        self.size = 0

    def enqueue(self, value):
        if self.is_full():
            raise OverflowError("Queue is full")
        self.rear = (self.rear + 1) % self.capacity
        self.array[self.rear] = value
        self.size += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        val = self.array[self.front]
        self.array[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return val

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.capacity

    def __repr__(self):
        return f"<QueueArray front={self.front}, rear={self.rear}, size={self.size}, array={self.array}>"


class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class QueueLinkedList:
    def __init__(self):
        self.front = None
        self.rear = None

    def enqueue(self, value):
        new_node = Node(value)
        if self.rear is None:
            self.front = self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node

    def dequeue(self):
        if self.front is None:
            raise IndexError("Dequeue from empty queue")
        val = self.front.value
        self.front = self.front.next
        if self.front is None:
            self.rear = None
        return val

    def is_empty(self):
        return self.front is None

    def __repr__(self):
        values = []
        temp = self.front
        while temp:
            values.append(str(temp.value))
            temp = temp.next
        return "<QueueLinkedList [" + " -> ".join(values) + "]>"

def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def quick_sort(arr):
    def partition(low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i+1], arr[high] = arr[high], arr[i+1]
        return i+1

    def quick_sort_recursive(low, high):
        if low < high:
            pi = partition(low, high)
            quick_sort_recursive(low, pi-1)
            quick_sort_recursive(pi+1, high)

    quick_sort_recursive(0, len(arr)-1)
    return arr


def heap_sort(arr):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

def heapify(arr, n, i):
    largest = i
    left = 2*i + 1
    right = 2*i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


def linear_search(arr, target):
    for idx, val in enumerate(arr):
        if val == target:
            return idx
    return -1

def binary_search(arr, target):
    low, high = 0, len(arr)-1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def measure_performance(func, *args, **kwargs):
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)
    start_time = time.perf_counter()

    result = func(*args, **kwargs)

    end_time = time.perf_counter()
    mem_after = process.memory_info().rss / (1024 * 1024)

    return {
        'time': end_time - start_time,
        'memory': mem_after - mem_before,
        'result': result
    }

def run_graph_demo():
    g = DependencyGraph()

    print("\nGraph Demo")
    try:
        num_tasks = int(input("Enter number of tasks to add: "))
        for _ in range(num_tasks):
            t = input("  Enter task name: ")
            g.add_task(t)

        num_deps = int(input("Enter number of dependencies: "))
        print("Format: 'Task1 Task2' means: Task2 depends on Task1.")
        for _ in range(num_deps):
            line = input("  Enter dependency (from to): ")
            parts = line.split()
            if len(parts) != 2:
                raise ValueError("Invalid dependency format")
            g.add_dependency(parts[0], parts[1])

        has_cycle = g.detect_cycle_dfs()
        if has_cycle:
            print("Cycle detected. Topological sort is not possible.")
        else:
            topo_order = g.topological_sort()
            print("No cycle detected.")
            print("Topological order of tasks:", topo_order)

        draw_choice = input("Draw graph? (y/n): ").lower()
        if draw_choice == 'y':
            g.draw_graph()

    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"Error: {e}")


def run_stack_queue_demo():
    print("\nStack & Queue Demo")
    try:
        capacity = int(input("Capacity for two-stacks array? (default=10): ") or 10)
        ts = TwoStacksOneArray(capacity)

        ts.push1("A1")
        ts.push1("A2")
        ts.push2("B1")
        ts.push2("B2")
        print("After some pushes:", ts)
        print("Pop from stack1:", ts.pop1())
        print("Pop from stack2:", ts.pop2())
        print("Current two-stacks state:", ts)

    except OverflowError as e:
        print(f"OverflowError: {e}")
    except IndexError as e:
        print(f"IndexError: {e}")

    try:
        qa = QueueArray(capacity=5)
        qa.enqueue("Q1")
        qa.enqueue("Q2")
        qa.enqueue("Q3")
        print("QueueArray after enqueues:", qa)
        print("Dequeue from QueueArray:", qa.dequeue())
        print("QueueArray after one dequeue:", qa)

    except (OverflowError, IndexError) as e:
        print(f"QueueArray Error: {e}")

    try:
        ql = QueueLinkedList()
        ql.enqueue("L1")
        ql.enqueue("L2")
        ql.enqueue("L3")
        print("QueueLinkedList after enqueues:", ql)
        print("Dequeue from QueueLinkedList:", ql.dequeue())
        print("QueueLinkedList after one dequeue:", ql)

    except IndexError as e:
        print(f"QueueLinkedList Error: {e}")


def run_sort_search_demo():
    print("\nSorting & Searching Demo")
    choice = input("Enter data manually (m) or generate random (r)? (m/r): ").lower()

    arr = []
    if choice == 'm':
        data_str = input("Enter numbers separated by spaces: ")
        arr = list(map(int, data_str.strip().split()))
    else:
        size = int(input("How many random integers? (default=10): ") or 10)
        arr = list(np.random.randint(0, 100, size))
        print("Generated array:", arr)

    print("\nChoose a sorting algorithm: ")
    print("1) Merge Sort")
    print("2) Quick Sort")
    print("3) Heap Sort")
    alg_choice = input("Enter choice: ").strip()

    arr_to_sort = arr[:]

    if alg_choice == '1':
        sorted_arr = measure_performance(merge_sort, arr_to_sort)
        print("Merge Sort => Time: {:.6f}s, MemChange: {:.3f}MB".format(
            sorted_arr['time'], sorted_arr['memory']))
        print("Sorted Result:", sorted_arr['result'])

    elif alg_choice == '2':
        sorted_arr = measure_performance(quick_sort, arr_to_sort)
        print("Quick Sort => Time: {:.6f}s, MemChange: {:.3f}MB".format(
            sorted_arr['time'], sorted_arr['memory']))
        print("Sorted Result:", sorted_arr['result'])

    elif alg_choice == '3':
        sorted_arr = measure_performance(heap_sort, arr_to_sort)
        print("Heap Sort => Time: {:.6f}s, MemChange: {:.3f}MB".format(
            sorted_arr['time'], sorted_arr['memory']))
        print("Sorted Result:", sorted_arr['result'])
    else:
        print("Invalid choice. Skipping sorting.")
        sorted_arr = {'result': arr_to_sort}

    search_choice = input("\nPerform searching? (y/n): ").lower()
    if search_choice == 'y':
        val_str = input("Enter value to search for: ")
        try:
            target_val = int(val_str)
            srch_on_sorted = input("Search in sorted array? (y/n): ").lower() == 'y'
            search_array = sorted_arr['result'] if srch_on_sorted else arr

            search_alg = input("Choose search [linear/binary]: ").lower()
            if search_alg == 'binary':
                idx = binary_search(search_array, target_val)
                print(f"Binary Search Index: {idx} ( -1 if not found )")
            else:
                idx = linear_search(search_array, target_val)
                print(f"Linear Search Index: {idx} ( -1 if not found )")

        except ValueError:
            print("Invalid integer for search.")
    else:
        print("Skipping searching.")


def main_menu():
    while True:
        print("\nMAIN MENU")
        print("1) Graph-Based Task Management")
        print("2) Stack & Queue Operations")
        print("3) Sorting & Searching")
        print("4) Exit")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            run_graph_demo()
        elif choice == '2':
            run_stack_queue_demo()
        elif choice == '3':
            run_sort_search_demo()
        elif choice == '4':
            print("Exiting. See ya.")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main_menu()