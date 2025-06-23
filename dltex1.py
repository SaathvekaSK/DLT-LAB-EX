def mcp_neuron(inputs, weights, threshold):
    total_input = sum(i * w for i, w in zip(inputs, weights))
    return 1 if total_input >= threshold else 0

def AND_gate(x1, x2):
    return mcp_neuron([x1, x2], [1, 1], 2)

def OR_gate(x1, x2):
    return mcp_neuron([x1, x2], [1, 1], 1)

def NOT_gate(x):
    return mcp_neuron([x], [-1], 0)

def NOR_gate(x1, x2):
    return mcp_neuron([x1, x2], [-1, -1], 0)

def XOR_gate(x1, x2):
    part1 = AND_gate(x1, NOT_gate(x2))
    part2 = AND_gate(NOT_gate(x1), x2)
    return OR_gate(part1, part2)

if __name__ == "__main__":
    print("AND(1,1):", AND_gate(1, 1))
    print("AND(1,0):", AND_gate(1, 0))
    print("OR(0,1):", OR_gate(0, 1))
    print("OR(0,0):", OR_gate(0, 0))
    print("NOT(0):", NOT_gate(0))
    print("NOT(1):", NOT_gate(1))
    print("NOR(0,0):", NOR_gate(0, 0))
    print("NOR(0,1):", NOR_gate(0, 1))
    print("XOR(0,0):", XOR_gate(0, 0))
    print("XOR(1,0):", XOR_gate(1, 0))
    print("XOR(0,1):", XOR_gate(0, 1))
    print("XOR(1,1):", XOR_gate(1, 1))
