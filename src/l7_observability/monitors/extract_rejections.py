import subprocess
import re

# Use tail to avoid memory issues with huge file
result = subprocess.run(
    ['tail', '-c', '500000', 'logs/trading_run_20260425T074834Z.log.archived'],
    capture_output=True,
    text=True,
    timeout=10
)

text = result.stdout
print("REJECTION AND EXECUTION ANALYSIS")
print("="*60)

# Find all EM:ZERO_AMT_BLOCK entries
zero_blocks = re.findall(r'\[EM:ZERO_AMT_BLOCK\].*', text)
print(f"\nFound {len(zero_blocks)} ZERO_AMT_BLOCK entries")
if zero_blocks:
    print("Sample:", zero_blocks[0][:150] if zero_blocks else "None")

# Find all Meta:Cooldown entries
cooldowns = re.findall(r'\[Meta:Cooldown\].*', text)
print(f"\nFound {len(cooldowns)} Cooldown entries")
if cooldowns:
    print("Sample:", cooldowns[0][:150] if cooldowns else "None")

# Find all LOOP_SUMMARY entries
loops = re.findall(r'LOOP_SUMMARY.*', text)
print(f"\nFound {len(loops)} LOOP_SUMMARY entries (in tail)")
if loops:
    print("\nLast 3 LOOP_SUMMARY entries:")
    for loop in loops[-3:]:
        # Extract key data
        m = re.search(r'loop_id=(\d+).*exec_result=([A-Z_]+)', loop)
        if m:
            print(f"  Loop {m.group(1)}: result={m.group(2)}")

