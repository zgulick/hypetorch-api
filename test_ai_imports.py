#!/usr/bin/env python3
"""
Test script to verify all AI system imports work correctly in the API environment.
"""

def test_ai_imports():
    """Test all AI system imports that the API might need."""
    
    print("🚀 Testing AI system imports for API deployment...")
    
    try:
        # Test basic AI modules
        from quality_gates import QualityGateManager
        print("✅ QualityGateManager imported")
        
        from adaptive_quality_manager import adaptive_quality_manager
        print("✅ adaptive_quality_manager imported")
        
        from confidence_scoring import calculate_context_confidence
        print("✅ confidence_scoring imported")
        
        from context_classifier import context_classifier
        print("✅ context_classifier imported") 
        
        from adaptive_context_scorer import adaptive_scorer
        print("✅ adaptive_context_scorer imported")
        
        from adaptive_ensemble_optimizer import ensemble_optimizer  
        print("✅ adaptive_ensemble_optimizer imported")
        
        from adaptive_hype_optimizer import hype_optimizer
        print("✅ adaptive_hype_optimizer imported")
        
        from validation_system import validation_system
        print("✅ validation_system imported")
        
        # Test API modules
        from api_v2 import v2_router
        print("✅ api_v2 imported")
        
        from api import app
        print("✅ main api imported")
        
        print("\n🎉 ALL AI SYSTEM IMPORTS SUCCESSFUL!")
        print("📦 API should now deploy without ModuleNotFoundError")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Other Error: {e}")
        return False

if __name__ == "__main__":
    success = test_ai_imports()
    exit(0 if success else 1)