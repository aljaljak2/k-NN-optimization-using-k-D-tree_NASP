# Implementation Notes: Bentley (1975)

**Paper**: "Multidimensional binary search trees used for associative searching"
**Source**: Communications of the ACM, Vol. 18, Issue 9, pp. 509-517
**Link**: https://dl.acm.org/doi/10.1145/361002.361007

## Key Concepts

### k-d Tree Structure
- Binary tree for k-dimensional data
- Each node contains:
  - `POINT`: k-dimensional record
  - `DISC`: discriminator (0 to k-1)
  - `LOSON`: left subtree (lesser values)
  - `HISON`: right subtree (greater values)

### Algorithms Implemented

#### 1. Algorithm INSERT (Section 3)
**Location**: `kdtree.cpp` lines 207-245

Steps:
- **I1**: Check if tree is empty
- **I2**: Compare with current node
- **I3**: Move down the tree
- **I4**: Insert as new leaf

**Complexity**: O(log n) average, O(n) worst case

#### 2. Algorithm DELETE (Section 4)
**Location**: `kdtree.cpp` lines 125-165

Steps:
- **D1**: Check if node is leaf
- **D2**: Decide replacement strategy
- **D3**: Find min in HISON subtree
- **D4**: Find max in LOSON subtree (if no HISON)

**Complexity**: O(log n) average

#### 3. SEARCH Operation
**Location**: `kdtree.cpp` lines 168-183

Traverses tree using discriminator comparison.

**Complexity**: O(log n) average

### Helper Functions

#### NEXTDISC
**Location**: `kdtree.cpp` lines 29-31
```cpp
int nextdisc(int disc) {
    return (disc + 1) % k;
}
```
Cycles through discriminators: 0, 1, ..., k-1, 0, ...

#### SUPERKEY
**Location**: `kdtree.cpp` lines 34-44

Creates cyclic concatenation: Kj, Kj+1, ..., Kk-1, K0, ..., Kj-1

Used for tie-breaking when coordinates are equal.

#### SUCCESSOR
**Location**: `kdtree.cpp` lines 49-69

Returns: LOSON, HISON, or EQUAL based on comparison.

## Implementation Decisions

1. **Memory Management**: Using raw pointers as in original paper, but with proper destructors
2. **Data Type**: `std::vector<double>` for k-dimensional points
3. **Comparison**: Lexicographic ordering via superkey for duplicate coordinates

## Testing

Test cases based on Figure 1 in the paper:
- 2D points: A(50,50), B(10,70), C(80,85), D(25,20), E(40,85), F(70,85)
- Verifies correct tree structure and operations

## Nearest Neighbor Extension

The paper briefly mentions nearest neighbor search. Our implementation adds:
- **Function**: `nearestNeighbor()` - lines 263-272
- **Helper**: `nearestNeighborRec()` - lines 284-305

This is a standard extension using:
- Best-first traversal
- Pruning based on distance bounds
- **Complexity**: O(log n) average, O(n) worst case

## References to Paper Sections

- Section 2: Tree structure definition
- Section 3: Insertion algorithm
- Section 4: Deletion algorithm
- Section 5: Partial match queries (not yet implemented)
- Section 6: Analysis and complexity
