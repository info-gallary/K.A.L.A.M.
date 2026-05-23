import sys
import time

def print_section(title):
    print("\n" + "="*30)
    print(f"🔧 {title}")
    print("="*30 + "\n", flush=True)

if __name__ == '__main__':
    folder_path = sys.argv[1] if len(sys.argv) > 1 else "N/A"
    print(f"📂 Starting test for folder: {folder_path}", flush=True)
    
    # Test 1: Simple print lines
    print_section("Basic Prints")
    print("Hello, World!", flush=True)
    print("Line 1", flush=True)
    print("Line 2\nLine 3", flush=True)
    
    # Test 2: Multiple newlines
    print_section("Multiple Newlines")
    print("Line before newlines\n\n\nLine after 3 newlines", flush=True)

    # Test 3: Tabs and spacing
    print_section("Tabs and Indentation")
    print("Item\tQuantity\tPrice", flush=True)
    print("Apple\t10\t₹50", flush=True)
    print("    This is indented using 4 spaces", flush=True)

    # Test 4: Emojis and Unicode
    print_section("Emojis and Unicode")
    print("✅ Task completed!", flush=True)
    print("⚠️ Warning: Something might be wrong", flush=True)
    print("❌ Error: Something went wrong", flush=True)
    print("🔥 Process started\n⏳ Waiting...\n🚀 Launching\n", flush=True)

    # Test 5: Delayed outputs
    print_section("Live/Streaming Effect")
    for i in range(3):
        print(f"⏱️ Step {i+1}/3 in progress...", flush=True)
        time.sleep(1)
    print("🎉 Steps completed!", flush=True)

    # Test 6: Errors to stderrc
    print_section("STDERR Test (shown in red in some terminals)")
    print("This is stdout", flush=True)
    print("This is stderr", file=sys.stderr, flush=True)

    # End
    print_section("End of Test")
    print("🧪 Test script finished", flush=True)
