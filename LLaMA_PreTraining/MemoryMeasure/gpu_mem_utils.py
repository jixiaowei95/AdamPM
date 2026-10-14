import torch


def _tensor_bytes(t: torch.Tensor):
    try:
        return t.numel() * t.element_size()
    except Exception:
        return 0

def model_parameters_bytes(model: torch.nn.Module):
    total = 0
    for p in model.parameters():
        total += _tensor_bytes(p)
    return total

def compute_and_print_weight_memory(model: torch.nn.Module, prefix: str = ""):
    weight_bytes = model_parameters_bytes(model)
    if prefix:
        print(f"\n{prefix} Memory of Weight: {weight_bytes / (1024**2):.4f} MiB ({weight_bytes} bytes)")
    return weight_bytes

def model_buffers_bytes(model: torch.nn.Module):
    total = 0
    for b in model.buffers():
        total += _tensor_bytes(b)
    return total

def model_grads_bytes(model: torch.nn.Module):
    total = 0
    for p in model.parameters():
        if p.grad is not None:
            total += _tensor_bytes(p.grad)
    return total



def _walk_and_collect_tensors(obj, tensors):
    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_and_collect_tensors(v, tensors)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _walk_and_collect_tensors(v, tensors)

def optimizer_state_bytes(optimizer: torch.optim.Optimizer, device: str = "cuda", include_cpu: bool = False):
    total = 0
    seen_storage_ptrs = set()
    
    for state in optimizer.state.values():
        for t in state.values():
            if not isinstance(t, torch.Tensor):
                continue
                
            if t.is_cuda and device != "cuda":
                continue
            if not t.is_cuda and device == "cuda" and not include_cpu:
                continue
                
            st = t.untyped_storage() if hasattr(t, 'untyped_storage') else t.storage()
            ptr = st.data_ptr()
            
            if ptr not in seen_storage_ptrs:
                seen_storage_ptrs.add(ptr)
                if hasattr(t, 'untyped_storage'):
                    total += st.size()
                else:
                    total += st.size() * t.element_size()
                    
    return total





def _estimate_obj_size_bytes(obj):
    total = 0
    if isinstance(obj, torch.Tensor):
        try:
            total += obj.numel() * obj.element_size()
        except Exception:
            pass
    elif isinstance(obj, dict):
        for v in obj.values():
            total += _estimate_obj_size_bytes(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            total += _estimate_obj_size_bytes(v)
    return total

def estimate_optimizer_state_bytes(optimizer):
    total_bytes = 0
    try:
        for param_key, state in optimizer.state.items():
            total_bytes += _estimate_obj_size_bytes(state)
    except Exception:
        return 0
    return total_bytes







def print_cuda_memory_breakdown(model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, prefix: str = ""):
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        mem_alloc_b = torch.cuda.memory_allocated(dev)
        mem_reserved_b = torch.cuda.memory_reserved(dev)
        peak_alloc_b = torch.cuda.max_memory_allocated(dev)
        peak_reserved_b = torch.cuda.max_memory_reserved(dev)
    else:
        mem_alloc_b = mem_reserved_b = peak_alloc_b = peak_reserved_b = 0

    params_b = model_parameters_bytes(model)
    buffers_b = model_buffers_bytes(model)
    grads_b = model_grads_bytes(model)
    opt_b = optimizer_state_bytes(optimizer) if optimizer is not None else 0

    other_b = mem_alloc_b - (params_b + buffers_b + grads_b + opt_b)


    print(f"\n{prefix}=== Memory ===")
    print(f"{prefix}Torch CUDA allocated : {mem_alloc_b / (1024**3):.4f} GiB")
    print(f"{prefix}Torch CUDA reserved  : {mem_reserved_b / (1024**3):.4f} GiB (cached by allocator)")
    print(f"{prefix}peak allocated       : {peak_alloc_b / (1024**3):.4f} GiB")
    print(f"{prefix}peak reserved        : {peak_reserved_b / (1024**3):.4f} GiB")
    print(f"{prefix}optimizer states     : {opt_b / (1024**2):.4f} MiB ({opt_b} bytes)")
    print(f"{prefix}model parameters     : {params_b / (1024**2):.4f} MiB ({params_b} bytes)")
    print(f"{prefix}model buffers        : {buffers_b / (1024**2):.4f} MiB")
    print(f"{prefix}grads (present)      : {grads_b / (1024**2):.4f} MiB")
    print(f"{prefix}other/temporary (approx): {other_b / (1024**2):.4f} MiB\n")






def optimizer_state_v_bytes(optimizer: torch.optim.Optimizer, device: str = "cuda", include_cpu: bool = False):
    total_v_bytes = 0
    seen_storage_ptrs = set()
    
    for state in optimizer.state.values():
        if 'v' in state:
            v_tensor = state['v']
        elif 'exp_avg_sq' in state:
            v_tensor = state['exp_avg_sq']
        else:
            continue
            
        if not isinstance(v_tensor, torch.Tensor):
            continue

        if v_tensor.is_cuda and device != "cuda":
            continue
        if not v_tensor.is_cuda and device == "cuda" and not include_cpu:
            continue

        st = v_tensor.untyped_storage() if hasattr(v_tensor, 'untyped_storage') else v_tensor.storage()
        ptr = st.data_ptr()
        
        if ptr not in seen_storage_ptrs:
            seen_storage_ptrs.add(ptr)
            if hasattr(v_tensor, 'untyped_storage'):
                total_v_bytes += st.size()
            else:
                total_v_bytes += st.size() * v_tensor.element_size()
                    
    return total_v_bytes

def optimizer_state_m_bytes(optimizer: torch.optim.Optimizer, device: str = "cuda", include_cpu: bool = False):
    total_m_bytes = 0
    seen_storage_ptrs = set()
    
    for state in optimizer.state.values():
        if 'm' in state:
            m_tensor = state['m']
        elif 'exp_avg' in state:
            m_tensor = state['exp_avg']
        else:
            continue
            
        if not isinstance(m_tensor, torch.Tensor):
            continue

        if m_tensor.is_cuda and device != "cuda":
            continue
        if not m_tensor.is_cuda and device == "cuda" and not include_cpu:
            continue

        st = m_tensor.untyped_storage() if hasattr(m_tensor, 'untyped_storage') else m_tensor.storage()
        ptr = st.data_ptr()
        
        if ptr not in seen_storage_ptrs:
            seen_storage_ptrs.add(ptr)
            if hasattr(m_tensor, 'untyped_storage'):
                total_m_bytes += st.size()
            else:
                total_m_bytes += st.size() * m_tensor.element_size()
                    
    return total_m_bytes






def print_optimizer_state_breakdown(optimizer: torch.optim.Optimizer, prefix: str = ""):
    if optimizer is None:
        return
    
    state_breakdown = {}
    
    for param, state_dict in optimizer.state.items():
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                bytes_used = value.numel() * value.element_size()
                if key not in state_breakdown:
                    state_breakdown[key] = 0
                state_breakdown[key] += bytes_used
    
    print(f"{prefix}=== Memory of optimizer state ===")
    total_bytes = 0
    for key, bytes_used in state_breakdown.items():
        print(f"{prefix}  {key}: {bytes_used / (1024**2):.4f} MiB, ({bytes_used} bytes)")
        total_bytes += bytes_used
    
    print(f"{prefix}  Total: {total_bytes / (1024**2):.4f} MiB, ({total_bytes} bytes)")
    
    if 'v' in state_breakdown:
        v_bytes = state_breakdown['v']
        print(f"{prefix}  Second Moment: {v_bytes / (1024**2):.4f} MiB, ({v_bytes} bytes)")
