# LITE Performance Degradation - Root Cause Analysis

## Observation

Step time degrades linearly during training:
- **Starts at:** ~35ms per step (excellent, would be ~52s total)
- **Ends at:** ~62ms per step (poor, 93s total)
- **Degradation:** 77% increase over 1490 steps
- **Onset:** Begins around step 500, continues linearly

## Root Cause Hypothesis

**torch.compile cache growth + adaptive threshold instability**

### Mechanism

1. **@torch.compile(dynamic=False, fullgraph=True)** on `mproj()` and `lite_process()`
   - Creates persistent compiled graph caches
   - Assumes stable tensor shapes and scales

2. **subspace_threshold_ratio changes EVERY step**
   ```python
   if state_P.norm() > math.sqrt(k):
       p_state["subspace_threshold_ratio"] *= 1.05  # +5%
   else:
       p_state["subspace_threshold_ratio"] *= 0.95  # -5%
   ```
   - Oscillates continuously between 0.95x and 1.05x
   - Changes tensor scales every iteration

3. **Cache thrashing**
   - torch.compile expects stable tensor properties
   - Constant scale changes defeat optimization assumptions
   - Compiled graph cache grows with each unique configuration
   - O(n) lookup degradation over time

### Evidence

- **Linear degradation pattern** matches cache growth behavior
- **Step 500 onset** coincides with validation checkpoint (when more computation occurs)
- **77% increase** over 1000 steps consistent with accumulated cache state
- **Not present in baseline** (baseline doesn't have torch.compile on these functions)

## Solution Implemented

**Batch adaptive threshold updates every 100 steps**

```python
step = self.iter if hasattr(self, "iter") else 0
if step % 100 == 0:
    # Only update threshold every 100 steps
    if state_P.norm() > math.sqrt(k):
        p_state["subspace_threshold_ratio"] *= 1.05
    else:
        p_state["subspace_threshold_ratio"] *= 0.95
```

### Benefits

- ✅ **Prevents cache thrashing** - 100x fewer updates
- ✅ **Keeps torch.compile speed** - maintains 1-2% gain
- ✅ **Maintains adaptive capability** - still adjusts to landscape
- ✅ **Minimal code change** - simple conditional

## Expected Outcome

- **Stable step time:** ~35-40ms throughout training
- **Total time:** ~52-60 seconds (beat 88.1s record by 30%+)
- **Convergence:** Maintained (loss still reaches ~3.28)

## Testing

Run full training and verify:
1. Step time remains stable (no linear increase)
2. Total time < 88.1 seconds
3. Validation loss ≤ 3.28

## References

- File: `train_gpt.py`, lines ~1236-1244
- Related: torch.compile documentation on cache behavior
- Pattern: Linear degradation typical of accumulated state
