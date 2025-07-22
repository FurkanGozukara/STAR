"""
Temporal Consistency Manager for SeedVR2 Integration

This module provides advanced temporal consistency management for SeedVR2 models,
integrating with STAR's scene detection system and providing intelligent
temporal overlap, scene-aware chunk optimization, and professional
consistency validation.

Features:
- Scene-aware temporal processing
- Intelligent temporal overlap calculation
- Consistency validation and optimization
- Integration with STAR's scene detection
- Advanced chunk boundary optimization
- Temporal quality analysis and monitoring
"""

import time
import torch
import numpy as np
import logging
import cv2
from typing import Dict, Any, List, Tuple, Optional, Union, Generator
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TemporalChunk:
    """Represents a temporal chunk with consistency information."""
    chunk_id: int
    start_frame: int
    end_frame: int
    overlap_start: int
    overlap_end: int
    scene_id: Optional[int] = None
    scene_boundary: bool = False
    temporal_complexity: float = 0.0
    consistency_score: float = 0.0


@dataclass
class SceneTemporalInfo:
    """Temporal information for a video scene."""
    scene_id: int
    start_frame: int
    end_frame: int
    frame_count: int
    avg_motion: float
    temporal_complexity: float
    recommended_overlap: int
    scene_change_score: float


class TemporalConsistencyValidator:
    """Validates and ensures temporal consistency requirements."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger
        self.min_batch_size = 5  # SeedVR2 requirement for temporal consistency
        
    def validate_temporal_config(self, seedvr2_config) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate temporal consistency configuration.
        
        Args:
            seedvr2_config: SeedVR2 configuration object
            
        Returns:
            Tuple of (is_valid, error_messages, corrected_config)
        """
        
        errors = []
        corrections = {}
        
        # Validate batch size for temporal consistency
        if seedvr2_config.batch_size < self.min_batch_size:
            errors.append(f"Batch size {seedvr2_config.batch_size} is below minimum {self.min_batch_size} required for temporal consistency")
            corrections["batch_size"] = self.min_batch_size
            
            if self.logger:
                self.logger.warning(f"Correcting batch size from {seedvr2_config.batch_size} to {self.min_batch_size} for temporal consistency")
        
        # Validate temporal overlap
        if seedvr2_config.temporal_overlap >= seedvr2_config.batch_size:
            max_overlap = max(1, seedvr2_config.batch_size - 2)
            errors.append(f"Temporal overlap {seedvr2_config.temporal_overlap} must be less than batch size {seedvr2_config.batch_size}")
            corrections["temporal_overlap"] = max_overlap
            
            if self.logger:
                self.logger.warning(f"Correcting temporal overlap from {seedvr2_config.temporal_overlap} to {max_overlap}")
        
        # Validate temporal overlap minimum for consistency
        if seedvr2_config.temporal_overlap > 0 and seedvr2_config.temporal_overlap < 2:
            errors.append(f"Temporal overlap {seedvr2_config.temporal_overlap} should be at least 2 frames for effective temporal consistency")
            corrections["temporal_overlap"] = 2
            
            if self.logger:
                self.logger.warning(f"Correcting temporal overlap from {seedvr2_config.temporal_overlap} to 2 for better consistency")
        
        is_valid = len(errors) == 0
        
        if self.logger and is_valid:
            self.logger.info(f"✅ Temporal consistency validation passed: batch_size={seedvr2_config.batch_size}, overlap={seedvr2_config.temporal_overlap}")
        
        return is_valid, errors, corrections
    
    def get_optimal_temporal_settings(
        self, 
        total_frames: int, 
        scene_count: int = 1,
        available_vram_gb: float = 8.0,
        target_quality: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Get optimal temporal consistency settings based on video characteristics.
        
        Args:
            total_frames: Total number of frames in video
            scene_count: Number of detected scenes
            available_vram_gb: Available VRAM in GB
            target_quality: Target quality level (fast, balanced, quality)
            
        Returns:
            Dictionary with optimal settings
        """
        
        # Base settings based on VRAM
        if available_vram_gb >= 16:
            base_batch_size = 12
            base_overlap = 3
        elif available_vram_gb >= 12:
            base_batch_size = 10
            base_overlap = 3
        elif available_vram_gb >= 8:
            base_batch_size = 8
            base_overlap = 2
        else:
            base_batch_size = 6
            base_overlap = 2
        
        # Ensure minimum for temporal consistency
        batch_size = max(base_batch_size, self.min_batch_size)
        
        # Adjust for quality target
        if target_quality == "quality":
            batch_size = min(batch_size + 2, 16)  # Higher batch size for quality
            overlap = min(base_overlap + 1, batch_size - 2)
        elif target_quality == "fast":
            batch_size = max(batch_size - 1, self.min_batch_size)
            overlap = max(base_overlap - 1, 1)
        else:  # balanced
            overlap = base_overlap
        
        # Adjust for scene complexity
        if scene_count > 10:  # Many scenes - use smaller batches for scene boundaries
            batch_size = max(batch_size - 1, self.min_batch_size)
            overlap = min(overlap + 1, batch_size - 2)
        
        # Calculate processing efficiency
        step_size = batch_size - overlap if overlap > 0 else batch_size
        estimated_chunks = max(1, (total_frames + step_size - 1) // step_size)
        efficiency_score = total_frames / (estimated_chunks * batch_size) if estimated_chunks > 0 else 1.0
        
        return {
            "batch_size": batch_size,
            "temporal_overlap": overlap,
            "estimated_chunks": estimated_chunks,
            "step_size": step_size,
            "efficiency_score": efficiency_score,
            "processing_time_multiplier": 1.0 + (overlap / batch_size * 0.3) if overlap > 0 else 1.0,
            "recommended_for_vram": available_vram_gb,
            "quality_level": target_quality
        }


class SceneTemporalAnalyzer:
    """Analyzes temporal characteristics of video scenes."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger
    
    def analyze_scene_temporal_info(
        self, 
        scene_video_paths: List[str],
        util_extract_frames = None,
        logger: logging.Logger = None
    ) -> List[SceneTemporalInfo]:
        """
        Analyze temporal characteristics of video scenes.
        
        Args:
            scene_video_paths: List of scene video file paths
            util_extract_frames: Frame extraction utility function
            logger: Logger instance
            
        Returns:
            List of SceneTemporalInfo objects
        """
        
        scene_temporal_infos = []
        
        for scene_idx, scene_path in enumerate(scene_video_paths):
            try:
                if logger:
                    logger.info(f"Analyzing temporal characteristics of scene {scene_idx + 1}")
                
                # Extract basic scene info
                frame_count, fps, frame_files = util_extract_frames(
                    scene_path, 
                    temp_output_dir=None,  # Just get info, don't extract
                    logger=logger,
                    info_only=True
                )
                
                # Analyze motion and temporal complexity
                motion_score, complexity_score = self._analyze_scene_motion(scene_path, logger)
                
                # Calculate recommended overlap based on motion
                if complexity_score > 0.7:  # High complexity
                    recommended_overlap = 4
                elif complexity_score > 0.4:  # Medium complexity
                    recommended_overlap = 3
                else:  # Low complexity
                    recommended_overlap = 2
                
                scene_info = SceneTemporalInfo(
                    scene_id=scene_idx,
                    start_frame=sum(len(self._get_scene_frames(sp)) for sp in scene_video_paths[:scene_idx]),
                    end_frame=sum(len(self._get_scene_frames(sp)) for sp in scene_video_paths[:scene_idx + 1]) - 1,
                    frame_count=frame_count,
                    avg_motion=motion_score,
                    temporal_complexity=complexity_score,
                    recommended_overlap=recommended_overlap,
                    scene_change_score=1.0  # Always 1.0 at scene boundaries
                )
                
                scene_temporal_infos.append(scene_info)
                
                if logger:
                    logger.info(f"Scene {scene_idx + 1}: {frame_count} frames, motion={motion_score:.2f}, complexity={complexity_score:.2f}, recommended_overlap={recommended_overlap}")
                    
            except Exception as e:
                if logger:
                    logger.error(f"Failed to analyze scene {scene_idx + 1}: {e}")
                
                # Create fallback scene info
                scene_info = SceneTemporalInfo(
                    scene_id=scene_idx,
                    start_frame=scene_idx * 100,  # Estimate
                    end_frame=(scene_idx + 1) * 100 - 1,
                    frame_count=100,  # Estimate
                    avg_motion=0.5,  # Medium motion estimate
                    temporal_complexity=0.5,
                    recommended_overlap=3,  # Safe default
                    scene_change_score=1.0
                )
                scene_temporal_infos.append(scene_info)
        
        return scene_temporal_infos
    
    def _analyze_scene_motion(self, scene_path: str, logger: logging.Logger = None) -> Tuple[float, float]:
        """
        Analyze motion and temporal complexity in a scene.
        
        Args:
            scene_path: Path to scene video file
            logger: Logger instance
            
        Returns:
            Tuple of (motion_score, complexity_score)
        """
        
        try:
            cap = cv2.VideoCapture(scene_path)
            
            if not cap.isOpened():
                return 0.5, 0.5  # Default values
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_frames = min(frame_count, 30)  # Sample up to 30 frames
            
            prev_frame = None
            motion_scores = []
            
            for i in range(sample_frames):
                # Jump to evenly spaced frames
                frame_idx = int(i * frame_count / sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate optical flow magnitude
                    diff = cv2.absdiff(gray, prev_frame)
                    motion_score = np.mean(diff) / 255.0
                    motion_scores.append(motion_score)
                
                prev_frame = gray
            
            cap.release()
            
            if motion_scores:
                avg_motion = np.mean(motion_scores)
                motion_variance = np.var(motion_scores)
                complexity_score = min(avg_motion + motion_variance, 1.0)
                
                return float(avg_motion), float(complexity_score)
            else:
                return 0.5, 0.5
                
        except Exception as e:
            if logger:
                logger.warning(f"Failed to analyze motion in scene {scene_path}: {e}")
            return 0.5, 0.5
    
    def _get_scene_frames(self, scene_path: str) -> List[str]:
        """Get frame count for a scene (placeholder)."""
        try:
            cap = cv2.VideoCapture(scene_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return [f"frame_{i:06d}.png" for i in range(frame_count)]
        except:
            return [f"frame_{i:06d}.png" for i in range(100)]  # Fallback


class TemporalChunkOptimizer:
    """Optimizes temporal chunk boundaries for scene-aware processing."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger
    
    def optimize_temporal_chunks(
        self,
        total_frames: int,
        batch_size: int,
        temporal_overlap: int,
        scene_temporal_infos: List[SceneTemporalInfo] = None,
        enable_scene_awareness: bool = True
    ) -> List[TemporalChunk]:
        """
        Create optimized temporal chunks with scene awareness.
        
        Args:
            total_frames: Total number of frames
            batch_size: Batch size for processing
            temporal_overlap: Temporal overlap between chunks
            scene_temporal_infos: Scene temporal information
            enable_scene_awareness: Whether to consider scene boundaries
            
        Returns:
            List of optimized TemporalChunk objects
        """
        
        if self.logger:
            self.logger.info(f"Optimizing temporal chunks: {total_frames} frames, batch_size={batch_size}, overlap={temporal_overlap}")
        
        chunks = []
        step_size = batch_size - temporal_overlap if temporal_overlap > 0 else batch_size
        
        # Create basic chunks
        chunk_id = 0
        current_frame = 0
        
        while current_frame < total_frames:
            chunk_start = current_frame
            chunk_end = min(chunk_start + batch_size, total_frames)
            
            # Calculate overlap regions
            overlap_start = max(0, chunk_start - temporal_overlap) if chunk_id > 0 else 0
            overlap_end = min(total_frames, chunk_end + temporal_overlap)
            
            # Determine scene information
            scene_id = None
            scene_boundary = False
            temporal_complexity = 0.5  # Default
            
            if enable_scene_awareness and scene_temporal_infos:
                scene_id, scene_boundary, temporal_complexity = self._analyze_chunk_scene_context(
                    chunk_start, chunk_end, scene_temporal_infos
                )
            
            chunk = TemporalChunk(
                chunk_id=chunk_id,
                start_frame=chunk_start,
                end_frame=chunk_end,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
                scene_id=scene_id,
                scene_boundary=scene_boundary,
                temporal_complexity=temporal_complexity,
                consistency_score=0.0  # Will be calculated during processing
            )
            
            chunks.append(chunk)
            
            current_frame += step_size
            chunk_id += 1
        
        # Post-process chunks for scene boundaries
        if enable_scene_awareness and scene_temporal_infos:
            chunks = self._optimize_scene_boundaries(chunks, scene_temporal_infos)
        
        if self.logger:
            self.logger.info(f"Created {len(chunks)} optimized temporal chunks")
            
            # Log chunk summary
            for i, chunk in enumerate(chunks[:5]):  # Show first 5
                self.logger.info(f"Chunk {i}: frames {chunk.start_frame}-{chunk.end_frame}, "
                               f"overlap {chunk.overlap_start}-{chunk.overlap_end}, "
                               f"scene_id={chunk.scene_id}, boundary={chunk.scene_boundary}")
        
        return chunks
    
    def _analyze_chunk_scene_context(
        self, 
        chunk_start: int, 
        chunk_end: int, 
        scene_temporal_infos: List[SceneTemporalInfo]
    ) -> Tuple[Optional[int], bool, float]:
        """
        Analyze scene context for a chunk.
        
        Args:
            chunk_start: Chunk start frame
            chunk_end: Chunk end frame
            scene_temporal_infos: Scene temporal information
            
        Returns:
            Tuple of (scene_id, is_scene_boundary, temporal_complexity)
        """
        
        scene_id = None
        scene_boundary = False
        temporal_complexity = 0.5
        
        # Find which scene(s) this chunk overlaps
        overlapping_scenes = []
        for scene_info in scene_temporal_infos:
            if not (chunk_end <= scene_info.start_frame or chunk_start >= scene_info.end_frame):
                overlapping_scenes.append(scene_info)
        
        if overlapping_scenes:
            # If chunk spans multiple scenes, it's a scene boundary
            if len(overlapping_scenes) > 1:
                scene_boundary = True
                scene_id = overlapping_scenes[0].scene_id
                # Use maximum complexity from overlapping scenes
                temporal_complexity = max(scene.temporal_complexity for scene in overlapping_scenes)
            else:
                # Single scene
                scene_info = overlapping_scenes[0]
                scene_id = scene_info.scene_id
                temporal_complexity = scene_info.temporal_complexity
                
                # Check if chunk is near scene boundary
                scene_margin = 5  # frames
                if (chunk_start <= scene_info.start_frame + scene_margin or 
                    chunk_end >= scene_info.end_frame - scene_margin):
                    scene_boundary = True
        
        return scene_id, scene_boundary, temporal_complexity
    
    def _optimize_scene_boundaries(
        self, 
        chunks: List[TemporalChunk], 
        scene_temporal_infos: List[SceneTemporalInfo]
    ) -> List[TemporalChunk]:
        """
        Optimize chunk boundaries to align with scene boundaries where beneficial.
        
        Args:
            chunks: List of temporal chunks
            scene_temporal_infos: Scene temporal information
            
        Returns:
            Optimized list of temporal chunks
        """
        
        optimized_chunks = []
        
        for chunk in chunks:
            # If chunk crosses scene boundary, consider splitting or adjusting
            if chunk.scene_boundary and len(scene_temporal_infos) > 1:
                # For now, just increase overlap for scene boundary chunks
                enhanced_chunk = TemporalChunk(
                    chunk_id=chunk.chunk_id,
                    start_frame=chunk.start_frame,
                    end_frame=chunk.end_frame,
                    overlap_start=max(0, chunk.overlap_start - 2),  # Extended overlap
                    overlap_end=min(chunk.end_frame + 10, chunk.overlap_end + 2),
                    scene_id=chunk.scene_id,
                    scene_boundary=chunk.scene_boundary,
                    temporal_complexity=chunk.temporal_complexity,
                    consistency_score=chunk.consistency_score
                )
                optimized_chunks.append(enhanced_chunk)
                
                if self.logger:
                    self.logger.info(f"Enhanced overlap for scene boundary chunk {chunk.chunk_id}")
            else:
                optimized_chunks.append(chunk)
        
        return optimized_chunks


class TemporalConsistencyManager:
    """Main temporal consistency manager integrating all components."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger
        self.validator = TemporalConsistencyValidator(logger)
        self.scene_analyzer = SceneTemporalAnalyzer(logger)
        self.chunk_optimizer = TemporalChunkOptimizer(logger)
        
    def configure_temporal_processing(
        self,
        seedvr2_config,
        video_info: Dict[str, Any],
        scene_video_paths: List[str] = None,
        enable_scene_awareness: bool = True,
        util_extract_frames = None
    ) -> Tuple[bool, Dict[str, Any], List[TemporalChunk]]:
        """
        Configure comprehensive temporal processing for SeedVR2.
        
        Args:
            seedvr2_config: SeedVR2 configuration object
            video_info: Video information dictionary
            scene_video_paths: List of scene video paths (if scene splitting enabled)
            enable_scene_awareness: Whether to enable scene-aware processing
            util_extract_frames: Frame extraction utility
            
        Returns:
            Tuple of (success, enhanced_config, temporal_chunks)
        """
        
        try:
            total_frames = video_info.get('frame_count', 0)
            
            if self.logger:
                self.logger.info(f"🎬 Configuring temporal consistency for {total_frames} frames")
            
            # Step 1: Validate and correct temporal settings
            is_valid, errors, corrections = self.validator.validate_temporal_config(seedvr2_config)
            
            # Apply corrections if needed
            enhanced_config = {
                "batch_size": corrections.get("batch_size", seedvr2_config.batch_size),
                "temporal_overlap": corrections.get("temporal_overlap", seedvr2_config.temporal_overlap),
                "enable_frame_padding": seedvr2_config.enable_frame_padding,
                "color_correction": seedvr2_config.color_correction,
                "flash_attention": seedvr2_config.flash_attention,
                "original_batch_size": seedvr2_config.batch_size,
                "original_temporal_overlap": seedvr2_config.temporal_overlap,
                "validation_errors": errors,
                "corrections_applied": corrections
            }
            
            # Step 2: Analyze scene temporal characteristics (if scenes available)
            scene_temporal_infos = []
            if enable_scene_awareness and scene_video_paths and util_extract_frames:
                if self.logger:
                    self.logger.info(f"🎭 Analyzing {len(scene_video_paths)} scenes for temporal characteristics")
                
                scene_temporal_infos = self.scene_analyzer.analyze_scene_temporal_info(
                    scene_video_paths, util_extract_frames, self.logger
                )
                
                enhanced_config["scene_count"] = len(scene_temporal_infos)
                enhanced_config["scene_temporal_infos"] = scene_temporal_infos
            
            # Step 3: Get optimal settings based on analysis
            optimal_settings = self.validator.get_optimal_temporal_settings(
                total_frames=total_frames,
                scene_count=len(scene_temporal_infos),
                available_vram_gb=video_info.get('available_vram_gb', 8.0),
                target_quality=video_info.get('target_quality', 'balanced')
            )
            
            # Merge optimal settings with corrected config
            enhanced_config.update(optimal_settings)
            
            # Step 4: Create optimized temporal chunks
            temporal_chunks = self.chunk_optimizer.optimize_temporal_chunks(
                total_frames=total_frames,
                batch_size=enhanced_config["batch_size"],
                temporal_overlap=enhanced_config["temporal_overlap"],
                scene_temporal_infos=scene_temporal_infos,
                enable_scene_awareness=enable_scene_awareness
            )
            
            enhanced_config["temporal_chunks"] = temporal_chunks
            enhanced_config["total_chunks"] = len(temporal_chunks)
            
            if self.logger:
                self.logger.info(f"✅ Temporal consistency configured successfully:")
                self.logger.info(f"   📊 Batch size: {enhanced_config['batch_size']} (was {seedvr2_config.batch_size})")
                self.logger.info(f"   🔄 Temporal overlap: {enhanced_config['temporal_overlap']} frames")
                self.logger.info(f"   🎬 Total chunks: {enhanced_config['total_chunks']}")
                self.logger.info(f"   🎭 Scene awareness: {enable_scene_awareness}")
                self.logger.info(f"   ⚡ Processing efficiency: {enhanced_config['efficiency_score']:.2f}")
            
            return True, enhanced_config, temporal_chunks
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Failed to configure temporal processing: {e}")
            
            return False, {}, []
    
    def validate_temporal_consistency_during_processing(
        self,
        chunk_results: List[torch.Tensor],
        temporal_chunks: List[TemporalChunk],
        logger: logging.Logger = None
    ) -> Dict[str, Any]:
        """
        Validate temporal consistency during processing.
        
        Args:
            chunk_results: List of processed chunk tensors
            temporal_chunks: List of temporal chunk information
            logger: Logger instance
            
        Returns:
            Consistency analysis results
        """
        
        consistency_scores = []
        
        for i, (result_tensor, chunk_info) in enumerate(zip(chunk_results, temporal_chunks)):
            try:
                # Calculate basic consistency score based on frame similarity
                if i > 0 and temporal_chunks[i-1].temporal_overlap > 0:
                    prev_tensor = chunk_results[i-1]
                    overlap_frames = min(chunk_info.temporal_overlap, prev_tensor.shape[0], result_tensor.shape[0])
                    
                    if overlap_frames > 0:
                        # Compare overlapping frames
                        prev_overlap = prev_tensor[-overlap_frames:]
                        curr_overlap = result_tensor[:overlap_frames]
                        
                        # Calculate frame-wise similarity
                        similarity = torch.cosine_similarity(
                            prev_overlap.flatten(1), 
                            curr_overlap.flatten(1), 
                            dim=1
                        ).mean().item()
                        
                        consistency_scores.append(similarity)
                        chunk_info.consistency_score = similarity
                        
                        if logger:
                            logger.debug(f"Chunk {i} temporal consistency: {similarity:.3f}")
            
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to calculate consistency for chunk {i}: {e}")
                consistency_scores.append(0.5)  # Default score
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        
        analysis = {
            "overall_consistency_score": overall_consistency,
            "chunk_consistency_scores": consistency_scores,
            "temporal_quality": "excellent" if overall_consistency > 0.9 else 
                              "good" if overall_consistency > 0.8 else
                              "fair" if overall_consistency > 0.7 else "poor",
            "chunks_analyzed": len(consistency_scores),
            "temporal_stability": "stable" if np.std(consistency_scores) < 0.1 else "variable"
        }
        
        if logger:
            logger.info(f"🎯 Temporal consistency analysis: {analysis['temporal_quality']} "
                       f"(score: {overall_consistency:.3f}, stability: {analysis['temporal_stability']})")
        
        return analysis


def create_temporal_consistency_manager(logger: logging.Logger = None) -> TemporalConsistencyManager:
    """Factory function to create a temporal consistency manager."""
    return TemporalConsistencyManager(logger)


def format_temporal_consistency_info(config: Dict[str, Any], analysis: Dict[str, Any] = None) -> str:
    """Format temporal consistency information for display."""
    
    info_lines = ["🎬 Temporal Consistency Configuration:"]
    
    # Basic settings
    info_lines.append(f"• Batch Size: {config.get('batch_size', 'Unknown')} frames")
    info_lines.append(f"• Temporal Overlap: {config.get('temporal_overlap', 'Unknown')} frames")
    info_lines.append(f"• Total Chunks: {config.get('total_chunks', 'Unknown')}")
    
    # Efficiency info
    efficiency = config.get('efficiency_score', 0)
    if efficiency > 0:
        info_lines.append(f"• Processing Efficiency: {efficiency:.1%}")
    
    # Scene awareness
    scene_count = config.get('scene_count', 0)
    if scene_count > 0:
        info_lines.append(f"• Scene-Aware Processing: {scene_count} scenes detected")
    
    # Corrections applied
    corrections = config.get('corrections_applied', {})
    if corrections:
        info_lines.append("• Auto-Corrections Applied:")
        for param, value in corrections.items():
            info_lines.append(f"  - {param}: corrected to {value}")
    
    # Analysis results (if available)
    if analysis:
        info_lines.append("\n🎯 Temporal Quality Analysis:")
        info_lines.append(f"• Overall Quality: {analysis.get('temporal_quality', 'Unknown').title()}")
        info_lines.append(f"• Consistency Score: {analysis.get('overall_consistency_score', 0):.1%}")
        info_lines.append(f"• Temporal Stability: {analysis.get('temporal_stability', 'Unknown').title()}")
    
    return "\n".join(info_lines) 