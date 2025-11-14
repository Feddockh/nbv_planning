"""
Test script to verify motion planning installation and functionality.

Run this to ensure pybullet_planning is correctly installed and
the MotionPlanner interface works as expected.
"""

import sys
import numpy as np

def test_imports():
    """Test that all required modules can be imported."""
    print("="*80)
    print("TEST 1: Checking imports...")
    print("="*80)
    
    try:
        import pybullet_planning
        print("✓ pybullet_planning installed")
    except ImportError as e:
        print(f"✗ pybullet_planning not found: {e}")
        return False
    
    try:
        from bodies.planning import MotionPlanner, smart_moveto
        print("✓ MotionPlanner imported")
    except ImportError as e:
        print(f"✗ MotionPlanner import failed: {e}")
        return False
    
    try:
        from bodies.planning_examples import example_basic_planning
        print("✓ planning_examples imported")
    except ImportError as e:
        print(f"✗ planning_examples import failed: {e}")
        return False
    
    print("\n✓ All imports successful!\n")
    return True


def test_planner_initialization():
    """Test that MotionPlanner can be initialized."""
    print("="*80)
    print("TEST 2: Initializing MotionPlanner...")
    print("="*80)
    
    try:
        import pybullet as p
        from bodies.planning import MotionPlanner
        
        # Create a simple PyBullet environment
        physics_client = p.connect(p.DIRECT)  # Headless mode
        
        # Load a simple robot (UR5 for testing)
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
        
        # Create a mock robot object
        class MockRobot:
            def __init__(self, body_id):
                self.body = body_id
                self.controllable_joints = list(range(7))
                self.end_effector = 6
        
        mock_robot = MockRobot(robot_id)
        
        # Initialize planner
        planner = MotionPlanner(mock_robot, obstacles=[])
        
        print(f"\n✓ Planner initialized successfully!")
        print(f"  Robot ID: {planner.body_id}")
        print(f"  Movable joints: {planner.movable_joints}")
        
        # Clean up
        p.disconnect()
        
        print("\n✓ MotionPlanner initialization test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Planner initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_planning_functions():
    """Test that planning functions are available and callable."""
    print("="*80)
    print("TEST 3: Checking MotionPlanner methods...")
    print("="*80)
    
    try:
        from bodies.planning import MotionPlanner
        
        required_methods = [
            'plan_to_joint_config',
            'plan_to_ee_pose',
            'execute_path',
            'plan_and_execute',
            'check_collision'
        ]
        
        for method_name in required_methods:
            if hasattr(MotionPlanner, method_name):
                print(f"✓ {method_name} available")
            else:
                print(f"✗ {method_name} missing")
                return False
        
        print("\n✓ All required methods available!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Method check failed: {e}")
        return False


def test_pybullet_planning_functions():
    """Test that key pybullet_planning functions are accessible."""
    print("="*80)
    print("TEST 4: Checking pybullet_planning functions...")
    print("="*80)
    
    try:
        from pybullet_planning import (
            plan_joint_motion,
            get_joint_positions,
            set_joint_positions,
            get_movable_joints
        )
        
        print("✓ plan_joint_motion available")
        print("✓ get_joint_positions available")
        print("✓ set_joint_positions available")
        print("✓ get_movable_joints available")
        
        print("\n✓ pybullet_planning functions accessible!\n")
        return True
        
    except ImportError as e:
        print(f"\n✗ pybullet_planning function import failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("MOTION PLANNING INSTALLATION TEST SUITE")
    print("="*80 + "\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_planner_initialization),
        ("Method Test", test_planning_functions),
        ("pybullet_planning Test", test_pybullet_planning_functions),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Motion planning is ready to use!")
        print("\nNext steps:")
        print("  1. See PLANNING_GUIDE.md for usage documentation")
        print("  2. Check bodies/planning_examples.py for example code")
        print("  3. Start using MotionPlanner in your NBV planning!")
    else:
        print("✗ SOME TESTS FAILED - Please check the errors above")
        print("\nTroubleshooting:")
        print("  1. Ensure pybullet_planning is installed: pip list | grep pybullet")
        print("  2. Check that you're in the mengine conda environment")
        print("  3. Try reinstalling: pip uninstall pybullet-planning && pip install pybullet-planning")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
