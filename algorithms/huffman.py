import heapq
from collections import Counter, namedtuple

# Define a simple node class for our Huffman Tree
class Node(namedtuple("Node", ["char", "freq"])):
    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanTree:
    def __init__(self):
        self.codes = {}  # Dictionary to store the Huffman codes for each symbol

    def build_tree(self, frequency):
        # Priority queue for building the Huffman tree
        heap = [Node(char, freq) for char, freq in frequency.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            # Pop two nodes with lowest frequency
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            # Create a new internal node with combined frequency
            merged = Node(None, left.freq + right.freq)
            heapq.heappush(heap, (merged.freq, merged, left, right))
        
        # Final tree node contains the root of the Huffman Tree
        if heap:
            self._generate_codes(heap[0][1], "")

    def _generate_codes(self, node, code):
        if node is None:
            return
        
        if node.char is not None:
            # It's a leaf node, assign the current code
            self.codes[node.char] = code
            return
        
        # Traverse the left and right branches
        self._generate_codes(node[2], code + "0")  # Left branch
        self._generate_codes(node[3], code + "1")  # Right branch

    def encode(self, data):
        """Encodes data using the generated Huffman codes."""
        return ''.join(self.codes[char] for char in data)

    def decode(self, encoded_data):
        """Decodes encoded data using the Huffman tree."""
        decoded = []
        node = self.tree
        for bit in encoded_data:
            node = node[2] if bit == '0' else node[3]  # Traverse based on bit
            if node.char is not None:  # Leaf node found
                decoded.append(node.char)
                node = self.tree  # Restart from root
        return ''.join(decoded)

# Example usage
# Step 1: Calculate frequency of each symbol in the quantized data
# Assuming `quantized_dct_patches` is flattened to a 1D array for encoding
# quantized_data = quantized_dct_patches.flatten()
# frequency = Counter(quantized_data)

# # Step 2: Build Huffman Tree
# huffman_tree = HuffmanTree()
# huffman_tree.build_tree(frequency)

# # Step 3: Encode data
# encoded_data = huffman_tree.encode(quantized_data)
# print("Encoded data:", encoded_data)

# # To decode
# decoded_data = huffman_tree.decode(encoded_data)
