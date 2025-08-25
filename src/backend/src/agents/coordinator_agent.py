# agents/coordinator_agent.py
"""
Coordinator Agent for aggregating decisions from all other agents.
Uses a lightweight quantized LLM for intelligent decision fusion.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import time
from dataclasses import asdict
from collections import defaultdict

from .base_agent import BaseAgent, AgentResult, RiskLevel

class CoordinatorAgent(BaseAgent):
    """
    Central coordinator that aggregates results from all other agents
    and makes final fraud detection decisions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("CoordinatorAgent", config)
        
        # Coordination parameters
        self.agent_weights = config.get('agent_weights', {
            'TouchPatternAgent': 0.2,
            'TypingBehaviorAgent': 0.15,
            'VoiceCommandAgent': 0.2,
            'VisualAgent': 0.25,
            'AppUsageAgent': 0.1,
            'MovementAgent': 0.1
        })
        
        # Decision thresholds
        self.risk_thresholds = config.get('risk_thresholds', {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        })

        # LLM parameters (simplified - would use quantized model in production)
        self.use_llm_fusion = config.get('use_llm_fusion', False)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        # Allow single-agent decisions in lightweight/dev setups by default
        self.min_agents_required = config.get('min_agents_required', 1)

        # History tracking
        self.decision_history = []
        self.agent_performance = defaultdict(list)

        # Adaptive weighting
        self.enable_adaptive_weights = config.get('enable_adaptive_weights', True)
        self.adaptation_window = config.get('adaptation_window', 100)

        print(f"[{self.agent_name}] Initialized with weights: {self.agent_weights}")
    
    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        """
        Coordinate analysis across all agents and make final decision
        
        Args:
            data: Dictionary containing results from all agents
            
        Returns:
            AgentResult with final fraud detection decision
        """
        start_time = time.time()
        
        try:
            # Extract agent results
            agent_results = data.get('agent_results', {})
            
            if not agent_results:
                return self._create_error_result("No agent results provided", start_time)
            
            # Parse agent results first
            parsed_results = {}
            for agent_name, result_data in agent_results.items():
                if isinstance(result_data, AgentResult):
                    parsed_results[agent_name] = result_data
                elif isinstance(result_data, dict):
                    # Convert dict to AgentResult
                    parsed_results[agent_name] = AgentResult(
                        agent_name=result_data.get('agent_name', agent_name),
                        anomaly_score=result_data.get('anomaly_score', 0.0),
                        risk_level=RiskLevel(result_data.get('risk_level', 'low')),
                        confidence=result_data.get('confidence', 0.0),
                        features_used=result_data.get('features_used', []),
                        processing_time_ms=result_data.get('processing_time_ms', 0.0),
                        metadata=result_data.get('metadata', {})
                    )
            
            # Check if we have enough agents after parsing
            if len(parsed_results) < self.min_agents_required:
                # Fallback: when only a single agent is available, perform a simple average
                # to avoid blocking in minimal-dev environments.
                weighted_sum = 0.0
                total_weight = 0.0
                for agent_name, result in parsed_results.items():
                    w = self.agent_weights.get(agent_name, 0.1)
                    weighted_sum += result.anomaly_score * w
                    total_weight += w

                if total_weight <= 0:
                    # Emergency fallback if all weights are zero
                    avg_score = 0.5
                    total_weight = 1.0
                else:
                    avg_score = weighted_sum / total_weight
                
                final_risk = self._determine_risk_level(avg_score, parsed_results)
                overall_confidence = self._calculate_overall_confidence(parsed_results)

                processing_time = (time.time() - start_time) * 1000
                return AgentResult(
                    agent_name=self.agent_name,
                    anomaly_score=float(avg_score),
                    risk_level=final_risk,
                    confidence=float(overall_confidence),
                    features_used=[],
                    processing_time_ms=processing_time,
                    metadata={'fallback': 'single_agent_average', 'agents_used': list(parsed_results.keys())}
                )
            
            # Weighted fusion of anomaly scores
            weighted_score = self._calculate_weighted_score(parsed_results)
            
            # Confidence-based adjustment
            confidence_adjusted_score = self._apply_confidence_weighting(parsed_results, weighted_score)
            
            # Risk level consensus
            final_risk_level = self._determine_risk_level(confidence_adjusted_score, parsed_results)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(parsed_results)
            
            # LLM-based fusion (if enabled)
            if self.use_llm_fusion and overall_confidence > self.confidence_threshold:
                llm_result = self._llm_decision_fusion(parsed_results, confidence_adjusted_score)
                if llm_result:
                    confidence_adjusted_score = llm_result['adjusted_score']
                    final_risk_level = RiskLevel(llm_result['risk_level'])
            
            # Collect all features used
            all_features = []
            for result in parsed_results.values():
                all_features.extend(result.features_used)
            
            # Create decision metadata
            metadata = {
                'agents_used': list(parsed_results.keys()),
                'agent_scores': {name: result.anomaly_score for name, result in parsed_results.items()},
                'agent_confidences': {name: result.confidence for name, result in parsed_results.items()},
                'weighted_score': float(weighted_score),
                'confidence_adjusted_score': float(confidence_adjusted_score),
                'agent_weights_used': {name: self.agent_weights.get(name, 0.1) for name in parsed_results.keys()},
                'decision_factors': self._analyze_decision_factors(parsed_results),
                'total_processing_time': sum(result.processing_time_ms for result in parsed_results.values())
            }
            
            # Store decision for adaptive learning
            decision_record = {
                'timestamp': time.time(),
                'agent_results': {name: asdict(result) for name, result in parsed_results.items()},
                'final_score': confidence_adjusted_score,
                'final_risk': final_risk_level.value,
                'confidence': overall_confidence
            }
            self.decision_history.append(decision_record)
            
            # Adapt weights if enabled
            if self.enable_adaptive_weights:
                self._update_agent_weights(parsed_results, confidence_adjusted_score)
            
            processing_time = (time.time() - start_time) * 1000
            
            return AgentResult(
                agent_name=self.agent_name,
                anomaly_score=float(confidence_adjusted_score),
                risk_level=final_risk_level,
                confidence=float(overall_confidence),
                features_used=list(set(all_features)),
                processing_time_ms=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            return self._create_error_result(f"Coordination error: {str(e)}", start_time)
    
    def _calculate_weighted_score(self, agent_results: Dict[str, AgentResult]) -> float:
        """Calculate weighted average of agent anomaly scores"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for agent_name, result in agent_results.items():
            weight = self.agent_weights.get(agent_name, 0.1)
            weighted_sum += result.anomaly_score * weight
            total_weight += weight
        
        if total_weight <= 0:
            # Emergency fallback: if all weights are zero, return average
            scores = [result.anomaly_score for result in agent_results.values()]
            return float(np.mean(scores)) if scores else 0.5
        
        return weighted_sum / total_weight
    
    def _apply_confidence_weighting(self, agent_results: Dict[str, AgentResult], base_score: float) -> float:
        """Adjust score based on agent confidences"""
        confidence_weights = []
        confidence_scores = []
        
        for result in agent_results.values():
            confidence_weights.append(result.confidence)
            confidence_scores.append(result.anomaly_score)
        
        if not confidence_weights:
            return base_score
        
        # Weight by confidence
        weighted_confidence_score = np.average(confidence_scores, weights=confidence_weights)
        
        # Blend with base score
        confidence_factor = np.mean(confidence_weights)
        adjusted_score = confidence_factor * weighted_confidence_score + (1 - confidence_factor) * base_score
        
        return float(np.clip(adjusted_score, 0.0, 1.0))
    
    def _determine_risk_level(self, score: float, agent_results: Dict[str, AgentResult]) -> RiskLevel:
        """Determine final risk level based on score and agent consensus"""
        
        # Score-based risk level
        if score >= self.risk_thresholds['high']:
            score_risk = RiskLevel.HIGH
        elif score >= self.risk_thresholds['medium']:
            score_risk = RiskLevel.MEDIUM
        else:
            score_risk = RiskLevel.LOW
        
        # Agent consensus
        risk_votes = [result.risk_level for result in agent_results.values()]
        high_votes = sum(1 for risk in risk_votes if risk == RiskLevel.HIGH)
        medium_votes = sum(1 for risk in risk_votes if risk == RiskLevel.MEDIUM)
        
        # Consensus logic
        total_votes = len(risk_votes)
        if high_votes > total_votes * 0.5:  # Majority high risk
            consensus_risk = RiskLevel.HIGH
        elif (high_votes + medium_votes) > total_votes * 0.6:  # High + medium majority
            consensus_risk = RiskLevel.MEDIUM
        else:
            consensus_risk = RiskLevel.LOW
        
        # Final decision (err on the side of caution)
        if score_risk == RiskLevel.HIGH or consensus_risk == RiskLevel.HIGH:
            return RiskLevel.HIGH
        elif score_risk == RiskLevel.MEDIUM or consensus_risk == RiskLevel.MEDIUM:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _calculate_overall_confidence(self, agent_results: Dict[str, AgentResult]) -> float:
        """Calculate overall confidence in the decision"""
        confidences = [result.confidence for result in agent_results.values()]
        
        if not confidences:
            return 0.0
        
        # Use weighted average of confidences
        weights = [self.agent_weights.get(result.agent_name, 0.1) for result in agent_results.values()]
        try:
            if sum(weights) <= 0:
                # Fallback to simple mean to avoid zero-sum weight errors
                overall_confidence = float(np.mean(confidences))
            else:
                overall_confidence = float(np.average(confidences, weights=weights))
        except Exception:
            overall_confidence = float(np.mean(confidences))
        
        # Penalize if too few agents
        agent_penalty = min(len(agent_results) / 4.0, 1.0)  # Assume 4 agents is ideal
        
        return float(overall_confidence * agent_penalty)
    
    def _analyze_decision_factors(self, agent_results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """Analyze the key factors contributing to the decision"""
        factors = {
            'high_risk_agents': [],
            'low_confidence_agents': [],
            'processing_times': {},
            'feature_coverage': set()
        }
        
        for agent_name, result in agent_results.items():
            if result.risk_level == RiskLevel.HIGH:
                factors['high_risk_agents'].append(agent_name)
            
            if result.confidence < 0.5:
                factors['low_confidence_agents'].append(agent_name)
            
            factors['processing_times'][agent_name] = result.processing_time_ms
            factors['feature_coverage'].update(result.features_used)
        
        factors['feature_coverage'] = list(factors['feature_coverage'])
        return factors
    
    def _llm_decision_fusion(self, agent_results: Dict[str, AgentResult], base_score: float) -> Optional[Dict[str, Any]]:
        """
        Use LLM for intelligent decision fusion (simplified implementation)
        In production, this would use a quantized LLM for reasoning
        """
        try:
            # Simplified rule-based fusion (placeholder for LLM)
            
            # Create context for LLM
            context = {
                'agent_count': len(agent_results),
                'high_risk_agents': [name for name, result in agent_results.items() if result.risk_level == RiskLevel.HIGH],
                'avg_confidence': np.mean([result.confidence for result in agent_results.values()]),
                'score_variance': np.var([result.anomaly_score for result in agent_results.values()]),
                'base_score': base_score
            }
            
            # Simple rule-based adjustment (placeholder for LLM reasoning)
            adjusted_score = base_score
            
            # If high variance in scores, reduce confidence
            if context['score_variance'] > 0.3:
                adjusted_score *= 0.9
            
            # If multiple high-risk agents agree, increase score
            if len(context['high_risk_agents']) >= 2:
                adjusted_score = min(adjusted_score * 1.2, 1.0)
            
            # If low overall confidence, be more conservative
            if context['avg_confidence'] < 0.5:
                adjusted_score *= 0.8
            
            # Determine risk level
            if adjusted_score >= 0.8:
                risk_level = 'high'
            elif adjusted_score >= 0.5:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return {
                'adjusted_score': adjusted_score,
                'risk_level': risk_level,
                'reasoning': f"LLM fusion based on {len(agent_results)} agents"
            }
            
        except Exception as e:
            print(f"[{self.agent_name}] LLM fusion error: {e}")
            return None
    
    def _update_agent_weights(self, agent_results: Dict[str, AgentResult], final_score: float) -> None:
        """
        Adaptively update agent weights based on performance
        (Simplified implementation - would use more sophisticated methods in production)
        """
        try:
            # Track performance for each agent
            for agent_name, result in agent_results.items():
                # Simple performance metric: how close was the agent to the final decision
                performance = 1.0 - abs(result.anomaly_score - final_score)
                self.agent_performance[agent_name].append(performance)
                
                # Keep only recent performance history
                if len(self.agent_performance[agent_name]) > self.adaptation_window:
                    self.agent_performance[agent_name] = self.agent_performance[agent_name][-self.adaptation_window:]
            
            # Update weights based on recent performance
            if len(self.decision_history) > 20:  # Need sufficient history
                for agent_name in self.agent_weights.keys():
                    if agent_name in self.agent_performance:
                        recent_performance = self.agent_performance[agent_name][-20:]  # Last 20 decisions
                        if len(recent_performance) >= 10:
                            avg_performance = np.mean(recent_performance)
                            
                            # Adjust weight slightly based on performance
                            current_weight = self.agent_weights[agent_name]
                            adjustment = (avg_performance - 0.5) * 0.1  # Small adjustment
                            new_weight = np.clip(current_weight + adjustment, 0.05, 0.5)
                            
                            self.agent_weights[agent_name] = new_weight
                
                # Normalize weights
                total_weight = sum(self.agent_weights.values())
                if total_weight > 0:
                    for agent_name in self.agent_weights:
                        self.agent_weights[agent_name] /= total_weight
            
        except Exception as e:
            print(f"[{self.agent_name}] Weight update error: {e}")
    
    def get_decision_summary(self, num_recent: int = 10) -> Dict[str, Any]:
        """Get summary of recent decisions"""
        recent_decisions = self.decision_history[-num_recent:] if self.decision_history else []
        
        summary = {
            'total_decisions': len(self.decision_history),
            'recent_decisions': len(recent_decisions),
            'current_weights': self.agent_weights.copy(),
            'agent_performance': {
                name: {
                    'avg_performance': np.mean(scores[-20:]) if scores else 0.0,
                    'decisions_count': len(scores)
                }
                for name, scores in self.agent_performance.items()
            }
        }
        
        if recent_decisions:
            summary['recent_risk_distribution'] = {
                'high': sum(1 for d in recent_decisions if d['final_risk'] == 'high'),
                'medium': sum(1 for d in recent_decisions if d['final_risk'] == 'medium'),
                'low': sum(1 for d in recent_decisions if d['final_risk'] == 'low')
            }
            
            summary['recent_avg_confidence'] = np.mean([d['confidence'] for d in recent_decisions])
            summary['recent_avg_score'] = np.mean([d['final_score'] for d in recent_decisions])
        
        return summary

    # Adapter methods to satisfy BaseAgent abstract interface
    def capture_data(self, sensor_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Coordinator operates on agent result dicts; no raw capture."""
        return None

    def predict(self, features: np.ndarray) -> AgentResult:
        """Not applicable for CoordinatorAgent; use analyze() with agent_results dict."""
        return self._create_error_result("Predict interface not supported; use analyze()", time.time())

    def train_initial(self, training_data: List[np.ndarray]) -> bool:
        """Coordinator has no conventional training; return True as noop."""
        return True

    def incremental_update(self, new_data: List[np.ndarray], is_anomaly: List[bool] = None) -> bool:
        """No incremental updates for coordinator in this simplified implementation."""
        return True
    
    def save_model(self, filepath: str) -> bool:
        """Save coordinator state to file"""
        try:
            coordinator_data = {
                'agent_weights': self.agent_weights,
                'agent_performance': {name: scores for name, scores in self.agent_performance.items()},
                'decision_history': self.decision_history[-100:],  # Keep recent history
                'risk_thresholds': self.risk_thresholds,
                'config': {
                    'min_agents_required': self.min_agents_required,
                    'confidence_threshold': self.confidence_threshold,
                    'enable_adaptive_weights': self.enable_adaptive_weights
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(coordinator_data, f, default=str, indent=2)
            
            print(f"[{self.agent_name}] State saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"[{self.agent_name}] Save error: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load coordinator state from file"""
        try:
            with open(filepath, 'r') as f:
                coordinator_data = json.load(f)
            
            self.agent_weights = coordinator_data['agent_weights']
            self.agent_performance = defaultdict(list, coordinator_data['agent_performance'])
            self.decision_history = coordinator_data['decision_history']
            self.risk_thresholds = coordinator_data['risk_thresholds']
            
            print(f"[{self.agent_name}] State loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"[{self.agent_name}] Load error: {e}")
            return False
    
    def _create_error_result(self, error_message: str, start_time: float) -> AgentResult:
        """Create an error result"""
        processing_time = (time.time() - start_time) * 1000
        return AgentResult(
            agent_name=self.agent_name,
            anomaly_score=1.0,
            risk_level=RiskLevel.HIGH,
            confidence=0.0,
            features_used=[],
            processing_time_ms=processing_time,
            metadata={'error': error_message}
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the coordinator state"""
        return {
            'agent_name': self.agent_name,
            'agent_weights': self.agent_weights,
            'decisions_made': len(self.decision_history),
            'adaptive_weights_enabled': self.enable_adaptive_weights,
            'agent_performance_tracked': len(self.agent_performance),
            'risk_thresholds': self.risk_thresholds,
            'min_agents_required': self.min_agents_required
        }
