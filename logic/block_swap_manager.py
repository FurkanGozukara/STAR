"""
Block Swap Manager for SeedVR2 Integration

This module provides advanced block swap management for SeedVR2 models,
including real-time memory monitoring, intelligent block selection,
and performance optimization for VRAM-limited systems.

Features:
- Real-time VRAM and system memory monitoring
- Intelligent block selection based on available memory
- Performance impact tracking and optimization
- Memory usage prediction and recommendations
- Advanced debugging and profiling capabilities
"""

import time
import torch
import psutil
import gc
import logging
import threading
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    vram_allocated_gb: float
    vram_reserved_gb: float
    vram_peak_gb: float
    system_ram_gb: float
    cpu_percent: float
    active_tensors: int


@dataclass 
class BlockSwapProfile:
    """Performance profile for block swap configuration."""
    blocks_swapped: int
    io_offloading: bool
    model_caching: bool
    vram_savings_gb: float
    performance_impact_percent: float
    memory_efficiency_score: float
    recommended_for_vram_gb: float


class MemoryMonitor:
    """Real-time memory monitoring for block swap optimization."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger
        self.is_monitoring = False
        self.snapshots: List[MemorySnapshot] = []
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 1.0  # seconds
        self.max_snapshots = 300  # 5 minutes at 1 second intervals
        
    def start_monitoring(self) -> None:
        """Start real-time memory monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.snapshots.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        if self.logger:
            self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        if self.logger:
            self.logger.info(f"Memory monitoring stopped. Captured {len(self.snapshots)} snapshots")
    
    def _monitor_loop(self) -> None:
        """Internal monitoring loop."""
        while self.is_monitoring:
            try:
                snapshot = self._capture_snapshot()
                self.snapshots.append(snapshot)
                
                # Limit snapshot history
                if len(self.snapshots) > self.max_snapshots:
                    self.snapshots.pop(0)
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Memory monitoring error: {e}")
            
            time.sleep(self.monitor_interval)
    
    def _capture_snapshot(self) -> MemorySnapshot:
        """Capture current memory state with multi-GPU support."""
        # Multi-GPU VRAM usage
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            
            # Aggregate VRAM across all GPUs
            total_allocated = 0
            total_reserved = 0
            total_peak = 0
            
            for i in range(device_count):
                try:
                    total_allocated += torch.cuda.memory_allocated(i)
                    total_reserved += torch.cuda.memory_reserved(i)
                    total_peak += torch.cuda.max_memory_allocated(i)
                except:
                    # Skip GPUs that can't be accessed
                    pass
            
            vram_allocated = total_allocated / (1024**3)
            vram_reserved = total_reserved / (1024**3)
            vram_peak = total_peak / (1024**3)
        else:
            vram_allocated = vram_reserved = vram_peak = 0.0
        
        # System memory
        system_memory = psutil.virtual_memory()
        system_ram_gb = system_memory.used / (1024**3)
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Active tensor count (expensive, so cache result)
        active_tensors = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])
        
        return MemorySnapshot(
            timestamp=time.time(),
            vram_allocated_gb=vram_allocated,
            vram_reserved_gb=vram_reserved,
            vram_peak_gb=vram_peak,
            system_ram_gb=system_ram_gb,
            cpu_percent=cpu_percent,
            active_tensors=active_tensors
        )
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage summary."""
        if not self.snapshots:
            snapshot = self._capture_snapshot()
        else:
            snapshot = self.snapshots[-1]
        
        return {
            "vram_allocated_gb": snapshot.vram_allocated_gb,
            "vram_reserved_gb": snapshot.vram_reserved_gb,
            "vram_peak_gb": snapshot.vram_peak_gb,
            "system_ram_gb": snapshot.system_ram_gb,
            "cpu_percent": snapshot.cpu_percent,
            "active_tensors": snapshot.active_tensors
        }
    
    def get_usage_trend(self, minutes: int = 2) -> Dict[str, Any]:
        """Analyze memory usage trend over specified time period."""
        if not self.snapshots:
            return {"status": "no_data"}
        
        cutoff_time = time.time() - (minutes * 60)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 2:
            return {"status": "insufficient_data"}
        
        # Calculate trends
        vram_values = [s.vram_allocated_gb for s in recent_snapshots]
        ram_values = [s.system_ram_gb for s in recent_snapshots]
        
        vram_trend = "stable"
        if vram_values[-1] > vram_values[0] * 1.1:
            vram_trend = "increasing"
        elif vram_values[-1] < vram_values[0] * 0.9:
            vram_trend = "decreasing"
        
        return {
            "status": "ok",
            "vram_trend": vram_trend,
            "vram_min": min(vram_values),
            "vram_max": max(vram_values),
            "vram_avg": sum(vram_values) / len(vram_values),
            "ram_avg": sum(ram_values) / len(ram_values),
            "samples": len(recent_snapshots)
        }


class BlockSwapOptimizer:
    """Intelligent block swap optimization and recommendations."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger
        self.profiles: List[BlockSwapProfile] = []
        self.model_profiles_cache: Dict[str, List[BlockSwapProfile]] = {}
        self._load_profiles()
    
    def _load_profiles(self) -> None:
        """Load block swap profiles from cache."""
        try:
            profiles_file = Path(__file__).parent / "block_swap_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    data = json.load(f)
                    self.model_profiles_cache = data
                    
                if self.logger:
                    self.logger.info(f"Loaded {len(self.model_profiles_cache)} model profiles")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to load block swap profiles: {e}")
    
    def save_profiles(self) -> None:
        """Save block swap profiles to cache."""
        try:
            profiles_file = Path(__file__).parent / "block_swap_profiles.json"
            with open(profiles_file, 'w') as f:
                json.dump(self.model_profiles_cache, f, indent=2)
                
            if self.logger:
                self.logger.info("Block swap profiles saved")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to save block swap profiles: {e}")
    
    def get_recommendations(self, 
                          available_vram_gb: float,
                          model_type: str = "3b",
                          target_quality: str = "balanced") -> Dict[str, Any]:
        """
        Get intelligent block swap recommendations.
        
        Args:
            available_vram_gb: Available VRAM in GB
            model_type: Model type (3b, 7b, etc.)
            target_quality: Target quality (fast, balanced, quality)
            
        Returns:
            Recommendations dictionary
        """
        
        # Base model VRAM requirements (estimated)
        model_vram_requirements = {
            "3b_fp8": 6.0,
            "3b_fp16": 8.0,
            "7b_fp8": 12.0,
            "7b_fp16": 16.0
        }
        
        model_key = f"{model_type}_fp8"  # Default to FP8
        base_requirement = model_vram_requirements.get(model_key, 8.0)
        
        # Calculate VRAM deficit
        vram_deficit = max(0, base_requirement - available_vram_gb)
        vram_ratio = available_vram_gb / base_requirement
        
        if vram_ratio >= 1.0:
            # Sufficient VRAM - block swap not needed
            return {
                "enable_block_swap": False,
                "block_swap_counter": 0,
                "offload_io": False,
                "model_caching": True,
                "reason": "Sufficient VRAM available",
                "vram_ratio": vram_ratio,
                "expected_performance": "optimal"
            }
        
        # Calculate recommended block swap configuration
        if vram_ratio >= 0.8:
            # Minimal block swap needed
            recommended_blocks = min(4, int(vram_deficit * 2))
            performance_impact = "minimal"
        elif vram_ratio >= 0.6:
            # Moderate block swap
            recommended_blocks = min(8, int(vram_deficit * 3))
            performance_impact = "moderate"
        else:
            # Aggressive block swap needed
            recommended_blocks = min(16, int(vram_deficit * 4))
            performance_impact = "significant"
        
        # I/O offloading recommendation
        offload_io = vram_ratio < 0.7
        
        # Model caching recommendation
        system_ram = psutil.virtual_memory().available / (1024**3)
        model_caching = system_ram > 8.0  # Only if sufficient RAM
        
        return {
            "enable_block_swap": True,
            "block_swap_counter": recommended_blocks,
            "offload_io": offload_io,
            "model_caching": model_caching,
            "reason": f"VRAM deficit: {vram_deficit:.1f}GB",
            "vram_ratio": vram_ratio,
            "expected_performance": performance_impact,
            "estimated_vram_savings": recommended_blocks * 0.3,  # Rough estimate
            "alternative_models": self._get_alternative_models(available_vram_gb)
        }
    
    def _get_alternative_models(self, available_vram_gb: float) -> List[str]:
        """Suggest alternative models that might work better."""
        alternatives = []
        
        if available_vram_gb >= 12:
            alternatives.append("7B FP8 (Highest Quality)")
        if available_vram_gb >= 8:
            alternatives.append("3B FP16 (High Quality)")
        if available_vram_gb >= 6:
            alternatives.append("3B FP8 (Balanced)")
        
        return alternatives
    
    def estimate_performance_impact(self, blocks_to_swap: int, io_offload: bool = False) -> Dict[str, float]:
        """Estimate performance impact of block swap configuration."""
        
        # Base performance impact estimates (empirical)
        base_impact_per_block = 2.5  # ~2.5% slowdown per block
        io_offload_impact = 5.0 if io_offload else 0.0
        
        total_impact = (blocks_to_swap * base_impact_per_block) + io_offload_impact
        
        # Memory savings estimate
        memory_savings_per_block = 0.3  # ~300MB per block
        total_memory_savings = blocks_to_swap * memory_savings_per_block
        
        # Efficiency score (higher is better)
        if total_impact > 0:
            efficiency_score = total_memory_savings / (total_impact / 100)
        else:
            efficiency_score = float('inf')
        
        return {
            "performance_impact_percent": min(total_impact, 80),  # Cap at 80%
            "memory_savings_gb": total_memory_savings,
            "efficiency_score": min(efficiency_score, 100),
            "recommendation": "optimal" if total_impact < 15 else "moderate" if total_impact < 35 else "aggressive"
        }


class BlockSwapManager:
    """Main block swap manager integrating monitoring and optimization."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger
        self.monitor = MemoryMonitor(logger)
        self.optimizer = BlockSwapOptimizer(logger)
        self.current_config: Optional[Dict[str, Any]] = None
        self.performance_stats: Dict[str, float] = {}
        
    def start_session(self) -> None:
        """Start a block swap optimization session."""
        self.monitor.start_monitoring()
        if self.logger:
            self.logger.info("Block swap session started")
    
    def end_session(self) -> None:
        """End the block swap session and save performance data."""
        self.monitor.stop_monitoring()
        
        # Analyze session performance
        self._analyze_session_performance()
        
        # Save optimizer profiles
        self.optimizer.save_profiles()
        
        if self.logger:
            self.logger.info("Block swap session ended")
    
    def configure_block_swap(self, seedvr2_config, force_analysis: bool = False) -> Dict[str, Any]:
        """
        Configure block swap with intelligent recommendations.
        
        Args:
            seedvr2_config: SeedVR2 configuration object
            force_analysis: Force re-analysis even if already configured
            
        Returns:
            Enhanced block swap configuration
        """
        
        if self.current_config and not force_analysis:
            return self.current_config
        
        # Get current memory status
        current_usage = self.monitor.get_current_usage()
        available_vram = self._estimate_available_vram()
        
        # Get recommendations
        model_type = self._extract_model_type(seedvr2_config.model)
        recommendations = self.optimizer.get_recommendations(
            available_vram_gb=available_vram,
            model_type=model_type,
            target_quality="balanced"
        )
        
        # Merge with user settings
        user_config = {
            "enable_block_swap": seedvr2_config.enable_block_swap,
            "blocks_to_swap": seedvr2_config.block_swap_counter,
            "offload_io": seedvr2_config.block_swap_offload_io,
            "model_caching": seedvr2_config.block_swap_model_caching
        }
        
        # Create enhanced configuration
        enhanced_config = {
            **user_config,
            "recommendations": recommendations,
            "current_vram_usage": current_usage,
            "performance_estimate": self.optimizer.estimate_performance_impact(
                user_config["blocks_to_swap"], 
                user_config["offload_io"]
            ),
            "debug_enabled": self.logger and self.logger.level <= logging.DEBUG,
            "use_non_blocking": True,
            "memory_threshold": 0.85
        }
        
        self.current_config = enhanced_config
        
        if self.logger:
            self.logger.info(f"Block swap configured: {user_config['blocks_to_swap']} blocks, "
                           f"VRAM: {available_vram:.1f}GB, "
                           f"Expected impact: {enhanced_config['performance_estimate']['performance_impact_percent']:.1f}%")
        
        return enhanced_config
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time block swap status and memory information."""
        current_usage = self.monitor.get_current_usage()
        usage_trend = self.monitor.get_usage_trend(minutes=2)
        
        status = {
            "memory_usage": current_usage,
            "usage_trend": usage_trend,
            "monitoring_active": self.monitor.is_monitoring,
            "snapshots_count": len(self.monitor.snapshots)
        }
        
        if self.current_config:
            status["block_swap_config"] = self.current_config
            status["performance_stats"] = self.performance_stats
        
        return status
    
    def _estimate_available_vram(self) -> float:
        """Estimate available VRAM for processing."""
        if not torch.cuda.is_available():
            return 0.0
        
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        current_usage = self.monitor.get_current_usage()
        
        # Reserve some VRAM for overhead
        available = total_vram - current_usage["vram_allocated_gb"] - 1.0  # 1GB safety margin
        
        return max(0.0, available)
    
    def _extract_model_type(self, model_filename: str) -> str:
        """Extract model type from filename."""
        filename_lower = model_filename.lower()
        
        if "3b" in filename_lower:
            return "3b"
        elif "7b" in filename_lower:
            return "7b"
        else:
            return "3b"  # Default fallback
    
    def _analyze_session_performance(self) -> None:
        """Analyze performance of the current session."""
        if not self.monitor.snapshots:
            return
        
        # Calculate session statistics
        snapshots = self.monitor.snapshots
        
        vram_values = [s.vram_allocated_gb for s in snapshots]
        cpu_values = [s.cpu_percent for s in snapshots]
        
        self.performance_stats = {
            "session_duration": snapshots[-1].timestamp - snapshots[0].timestamp,
            "vram_peak": max(vram_values),
            "vram_avg": sum(vram_values) / len(vram_values),
            "cpu_avg": sum(cpu_values) / len(cpu_values),
            "memory_efficiency": min(vram_values) / max(vram_values) if max(vram_values) > 0 else 1.0
        }
        
        if self.logger:
            self.logger.info(f"Session performance: VRAM peak {self.performance_stats['vram_peak']:.1f}GB, "
                           f"avg {self.performance_stats['vram_avg']:.1f}GB, "
                           f"efficiency {self.performance_stats['memory_efficiency']:.2f}")


def create_block_swap_manager(logger: logging.Logger = None) -> BlockSwapManager:
    """Factory function to create a configured block swap manager."""
    return BlockSwapManager(logger)


def format_memory_info(memory_usage: Dict[str, float]) -> str:
    """Format memory usage information for display."""
    vram_info = f"VRAM: {memory_usage['vram_allocated_gb']:.1f}/{memory_usage['vram_reserved_gb']:.1f}GB"
    ram_info = f"RAM: {memory_usage['system_ram_gb']:.1f}GB"
    cpu_info = f"CPU: {memory_usage['cpu_percent']:.1f}%"
    
    return f"{vram_info} | {ram_info} | {cpu_info}"


def get_multi_gpu_utilization() -> Dict[str, Any]:
    """Get detailed multi-GPU utilization information."""
    
    if not torch.cuda.is_available():
        return {"available": False, "gpus": []}
    
    gpu_info = []
    device_count = torch.cuda.device_count()
    
    for i in range(device_count):
        try:
            # Basic CUDA info
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = props.total_memory / (1024**3)
            
            gpu_data = {
                "id": i,
                "name": props.name,
                "vram_allocated_gb": allocated,
                "vram_reserved_gb": reserved,
                "vram_total_gb": total,
                "vram_free_gb": total - allocated,
                "utilization_percent": 0,  # Default if NVML unavailable
                "temperature_c": None,
                "power_usage_w": None
            }
            
            # Try to get advanced metrics via NVML
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_data["utilization_percent"] = util.gpu
                gpu_data["memory_utilization_percent"] = util.memory
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_data["temperature_c"] = temp
                
                # Power usage
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                gpu_data["power_usage_w"] = power
                
            except:
                # NVML not available or error
                pass
            
            gpu_info.append(gpu_data)
            
        except Exception as e:
            # Add error entry for this GPU
            gpu_info.append({
                "id": i,
                "name": f"GPU {i}",
                "error": str(e),
                "available": False
            })
    
    return {
        "available": True,
        "device_count": device_count,
        "gpus": gpu_info,
        "total_vram_gb": sum(gpu.get("vram_total_gb", 0) for gpu in gpu_info),
        "total_allocated_gb": sum(gpu.get("vram_allocated_gb", 0) for gpu in gpu_info),
        "total_free_gb": sum(gpu.get("vram_free_gb", 0) for gpu in gpu_info)
    }


def monitor_multi_gpu_performance(gpu_devices: List[int], duration_seconds: int = 60) -> Dict[str, Any]:
    """Monitor multi-GPU performance over time."""
    
    if not torch.cuda.is_available():
        return {"available": False}
    
    snapshots = []
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        snapshot = {
            "timestamp": time.time(),
            "gpus": []
        }
        
        for gpu_id in gpu_devices:
            try:
                # Memory info
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                
                gpu_snapshot = {
                    "id": gpu_id,
                    "vram_allocated_gb": allocated,
                    "vram_reserved_gb": reserved,
                    "utilization_percent": None,
                    "temperature_c": None
                }
                
                # NVML metrics if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_snapshot["utilization_percent"] = util.gpu
                    
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_snapshot["temperature_c"] = temp
                    
                except:
                    pass
                
                snapshot["gpus"].append(gpu_snapshot)
                
            except Exception as e:
                snapshot["gpus"].append({
                    "id": gpu_id,
                    "error": str(e)
                })
        
        snapshots.append(snapshot)
        time.sleep(1)  # 1 second intervals
    
    # Analyze performance
    analysis = {
        "monitoring_duration": duration_seconds,
        "total_snapshots": len(snapshots),
        "gpu_performance": {}
    }
    
    for gpu_id in gpu_devices:
        gpu_snapshots = []
        for snapshot in snapshots:
            gpu_data = next((gpu for gpu in snapshot["gpus"] if gpu["id"] == gpu_id), None)
            if gpu_data and "error" not in gpu_data:
                gpu_snapshots.append(gpu_data)
        
        if gpu_snapshots:
            vram_values = [s["vram_allocated_gb"] for s in gpu_snapshots]
            util_values = [s["utilization_percent"] for s in gpu_snapshots if s["utilization_percent"] is not None]
            temp_values = [s["temperature_c"] for s in gpu_snapshots if s["temperature_c"] is not None]
            
            analysis["gpu_performance"][gpu_id] = {
                "vram_avg": sum(vram_values) / len(vram_values),
                "vram_peak": max(vram_values),
                "vram_min": min(vram_values),
                "utilization_avg": sum(util_values) / len(util_values) if util_values else 0,
                "utilization_peak": max(util_values) if util_values else 0,
                "temperature_avg": sum(temp_values) / len(temp_values) if temp_values else None,
                "temperature_peak": max(temp_values) if temp_values else None
            }
    
    return {
        "available": True,
        "snapshots": snapshots,
        "analysis": analysis
    }


def format_multi_gpu_status(multi_gpu_info: Dict[str, Any]) -> str:
    """Format multi-GPU status for display."""
    
    if not multi_gpu_info.get("available", False):
        return "Multi-GPU: Not Available"
    
    gpus = multi_gpu_info.get("gpus", [])
    if not gpus:
        return "Multi-GPU: No GPUs Detected"
    
    # Summary info
    total_vram = multi_gpu_info.get("total_vram_gb", 0)
    total_free = multi_gpu_info.get("total_free_gb", 0)
    device_count = len(gpus)
    
    # Count active GPUs (with utilization data)
    active_gpus = len([gpu for gpu in gpus if gpu.get("utilization_percent", 0) > 10])
    
    status_parts = [
        f"Multi-GPU: {device_count} devices",
        f"Total VRAM: {total_vram:.1f}GB",
        f"Free: {total_free:.1f}GB"
    ]
    
    if active_gpus > 0:
        status_parts.append(f"Active: {active_gpus} GPUs")
    
    # Add temperature warning if any GPU is hot
    max_temp = max([gpu.get("temperature_c", 0) for gpu in gpus if gpu.get("temperature_c") is not None], default=0)
    if max_temp > 80:
        status_parts.append(f"⚠️ Hot: {max_temp}°C")
    elif max_temp > 0:
        status_parts.append(f"Temp: {max_temp}°C")
    
    return " | ".join(status_parts)


def format_block_swap_status(config: Dict[str, Any]) -> str:
    """Format block swap configuration status for display."""
    if not config.get("enable_block_swap", False):
        return "Block Swap: Disabled"
    
    blocks = config.get("blocks_to_swap", 0)
    io_offload = config.get("offload_io", False)
    caching = config.get("model_caching", False)
    
    perf_est = config.get("performance_estimate", {})
    impact = perf_est.get("performance_impact_percent", 0)
    savings = perf_est.get("memory_savings_gb", 0)
    
    status_parts = [f"Block Swap: {blocks} blocks"]
    
    if io_offload:
        status_parts.append("I/O offload")
    if caching:
        status_parts.append("Model caching")
    
    status_parts.append(f"Impact: ~{impact:.1f}%")
    status_parts.append(f"Savings: ~{savings:.1f}GB")
    
    return " | ".join(status_parts) 