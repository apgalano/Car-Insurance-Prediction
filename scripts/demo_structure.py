#!/usr/bin/env python3
"""Demo script to show the new project structure."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    """Demonstrate the new project structure."""
    print("Car Insurance Prediction - New Project Structure Demo")
    print("=" * 55)
    
    # Show project structure
    project_root = Path(__file__).parent.parent
    
    print("\nProject Structure:")
    print(f"📁 {project_root.name}/")
    
    # List main directories
    for item in sorted(project_root.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            print(f"  📁 {item.name}/")
            
            # Show some subdirectories
            if item.name in ['src', 'data', 'outputs']:
                for subitem in sorted(item.iterdir()):
                    if subitem.is_dir():
                        print(f"    📁 {subitem.name}/")
                    elif subitem.suffix == '.py':
                        print(f"    📄 {subitem.name}")
    
    print("\nKey Improvements:")
    print("✅ Modular code organization")
    print("✅ Separation of concerns")
    print("✅ Configuration management")
    print("✅ Proper project structure")
    print("✅ Testing framework")
    print("✅ Documentation")
    
    print("\nTo run the full pipeline:")
    print("1. Set up virtual environment:")
    print("   ./setup.sh")
    print("2. Activate virtual environment:")
    print("   source venv/bin/activate")
    print("3. Run training: python scripts/train_model.py")
    
    print("\nStructure Benefits:")
    print("• Easy to maintain and extend")
    print("• Reusable components")
    print("• Better testing capabilities")
    print("• Professional code organization")


if __name__ == "__main__":
    main()