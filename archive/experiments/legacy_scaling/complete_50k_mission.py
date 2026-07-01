#!/usr/bin/env python3
"""
Complete 50K+ Mission Execution
===============================

Master script to execute all three strategies for achieving 50K+ samples:
1. Targeted expansion of high-yield cancer types
2. Clinical and miRNA data integration  
3. Extended TCGA sampling across all cancer types

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import logging
import time
from datetime import datetime
from pathlib import Path
import subprocess
import sys

# Setup comprehensive logging
log_file = f"complete_50k_mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Complete50KMission:
    """Master coordinator for achieving 50K+ samples"""
    
    def __init__(self):
        self.mission_start_time = datetime.now()
        self.strategies = [
            {
                'name': 'Targeted High-Yield Expansion',
                'script': 'targeted_high_yield_expansion.py',
                'description': 'Focus on KIRC, UCEC, OV for ~7500 samples',
                'estimated_samples': 7500,
                'priority': 1
            },
            {
                'name': 'Clinical & miRNA Integration',
                'script': 'clinical_mirna_integration.py', 
                'description': 'Add clinical/miRNA across all 33 cancer types',
                'estimated_samples': 3300,
                'priority': 2
            },
            {
                'name': 'Extended TCGA Sampling',
                'script': 'extended_tcga_sampling.py',
                'description': 'Comprehensive expansion of all cancer types',
                'estimated_samples': 15000,
                'priority': 3
            }
        ]
        
        self.total_estimated_boost = sum(s['estimated_samples'] for s in self.strategies)
        
    def execute_strategy(self, strategy):
        """Execute a single strategy"""
        
        name = strategy['name']
        script = strategy['script']
        estimated_samples = strategy['estimated_samples']
        
        logger.info(f"🚀 EXECUTING STRATEGY: {name}")
        logger.info(f"📊 Script: {script}")
        logger.info(f"🎯 Estimated sample boost: {estimated_samples:,}")
        logger.info("=" * 80)
        
        strategy_start_time = time.time()
        
        try:
            # Execute the strategy script
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout per strategy
            )
            
            strategy_duration = time.time() - strategy_start_time
            
            if result.returncode == 0:
                logger.info(f"✅ {name} COMPLETED SUCCESSFULLY!")
                logger.info(f"⏱️ Duration: {strategy_duration/60:.1f} minutes")
                logger.info(f"📋 Output preview: {result.stdout[-500:]}")  # Last 500 chars
                return True
            else:
                logger.error(f"❌ {name} FAILED!")
                logger.error(f"Error: {result.stderr[:500]}")  # First 500 chars of error
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {name} TIMED OUT (2 hours)")
            return False
        except Exception as e:
            logger.error(f"💥 {name} CRASHED: {e}")
            return False
    
    def run_final_integration(self):
        """Run final integration to count total samples achieved"""
        
        logger.info("\n🔄 RUNNING FINAL SAMPLE INTEGRATION...")
        logger.info("=" * 80)
        
        try:
            # Run the comprehensive integrator on all new data
            result = subprocess.run(
                [sys.executable, 'comprehensive_multi_omics_integrator.py'],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Final integration completed successfully!")
                
                # Run final validation
                validation_result = subprocess.run(
                    [sys.executable, 'final_50k_validation.py'],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if validation_result.returncode == 0:
                    logger.info("✅ Final validation completed!")
                    logger.info("📊 VALIDATION RESULTS:")
                    logger.info(validation_result.stdout)
                    return True
                else:
                    logger.error("❌ Final validation failed!")
                    logger.error(validation_result.stderr)
                    
            else:
                logger.error("❌ Final integration failed!")
                logger.error(result.stderr[:500])
                
        except Exception as e:
            logger.error(f"💥 Final integration crashed: {e}")
        
        return False
    
    def execute_complete_mission(self):
        """Execute the complete 50K+ achievement mission"""
        
        logger.info("🚀 STARTING COMPLETE 50K+ SAMPLE ACHIEVEMENT MISSION")
        logger.info("=" * 100)
        logger.info(f"📅 Mission Start Time: {self.mission_start_time}")
        logger.info(f"🎯 Current Status: 43,631 samples (87.3% of 50K target)")
        logger.info(f"🎯 Gap to Close: 6,369 samples")
        logger.info(f"📊 Strategies: {len(self.strategies)}")
        logger.info(f"🚀 Total Estimated Boost: {self.total_estimated_boost:,} samples")
        logger.info("=" * 100)
        
        successful_strategies = 0
        
        # Execute each strategy in priority order
        for strategy in sorted(self.strategies, key=lambda x: x['priority']):
            logger.info(f"\n📍 STRATEGY {strategy['priority']}: {strategy['name']}")
            
            success = self.execute_strategy(strategy)
            if success:
                successful_strategies += 1
                logger.info(f"✅ Strategy {strategy['priority']} completed successfully!")
            else:
                logger.warning(f"⚠️ Strategy {strategy['priority']} had issues but continuing...")
            
            # Brief pause between strategies
            time.sleep(30)
        
        # Execute final integration and validation
        logger.info("\n🏁 EXECUTING FINAL INTEGRATION & VALIDATION")
        logger.info("=" * 80)
        
        final_success = self.run_final_integration()
        
        # Calculate mission results
        mission_duration = datetime.now() - self.mission_start_time
        
        logger.info("\n" + "=" * 100)
        logger.info("🏆 COMPLETE 50K+ MISSION RESULTS")
        logger.info("=" * 100)
        logger.info(f"⏱️ Total Mission Duration: {mission_duration}")
        logger.info(f"✅ Successful Strategies: {successful_strategies}/{len(self.strategies)}")
        logger.info(f"🎯 Final Integration: {'SUCCESS' if final_success else 'ISSUES'}")
        
        if successful_strategies >= 2 and final_success:
            logger.info("🎉 MISSION STATUS: ✅ SUCCESS!")
            logger.info("🚀 50K+ sample target should be ACHIEVED!")
        elif successful_strategies >= 1:
            logger.info("🎯 MISSION STATUS: ⚠️ PARTIAL SUCCESS")
            logger.info("📊 Significant progress made toward 50K+ target")
        else:
            logger.info("❌ MISSION STATUS: 🔄 REQUIRES RETRY")
            logger.info("⚙️ Technical issues encountered - retry recommended")
        
        logger.info(f"📋 Detailed log saved to: {log_file}")
        logger.info("=" * 100)
        
        return successful_strategies, final_success

if __name__ == "__main__":
    mission = Complete50KMission()
    strategies_success, integration_success = mission.execute_complete_mission()
