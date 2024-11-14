import sys
sys.path.append('..')
from include.common_imports import *

class HuffmanTree:
    def __init__(self):
        self.tree = None
        self.codes = {}
        self.reverse_codes = {}

    class Node:
        def __init__(self, symbol, frequency):
            self.symbol = symbol
            self.frequency = frequency
            self.left = None
            self.right = None
        
        # Define comparison operators for the heap
        def __lt__(self, other):
            return self.frequency < other.frequency
        
        def __eq__(self, other):
            return self.frequency == other.frequency

    def build_tree(self, frequency_dict):
        # Create a priority queue (min-heap)
        heap = [self.Node(symbol, freq) for symbol, freq in frequency_dict.items()]
        heapq.heapify(heap)
        
        # Merge nodes until only one remains (the root of the tree)
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            # Create a new node with the combined frequency of left and right
            merged = self.Node(None, left.frequency + right.frequency)
            merged.left = left
            merged.right = right
            
            heapq.heappush(heap, merged)
        
        # The final element in the heap is the root of the Huffman tree
        self.tree = heap[0]
        
        # Generate codes for each symbol by traversing the tree
        self._generate_codes(self.tree)

    def _generate_codes(self, node, code=""):
        # Recursively traverse the tree to generate codes
        if node is not None:
            if node.symbol is not None:
                # Leaf node, add to codes
                self.codes[node.symbol] = code
                self.reverse_codes[code] = node.symbol
            self._generate_codes(node.left, code + "0")
            self._generate_codes(node.right, code + "1")

    def encode(self, data):
        # Encode data using the Huffman codes
        return ''.join(self.codes[symbol] for symbol in data)

    def decode(self, encoded_data):
        # Decode the Huffman-encoded data using the reverse_codes
        decoded_data = []
        current_code = ""
        # We need to populate reverse_codes too:
        for key, value in self.codes.items():
            self.reverse_codes[value] = key
        for bit in encoded_data:
            current_code += bit
            if current_code in self.reverse_codes:
                decoded_data.append(self.reverse_codes[current_code])
                current_code = ""
        
        return decoded_data
