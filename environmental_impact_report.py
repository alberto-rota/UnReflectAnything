import wandb
from dotenv import load_dotenv
import os
import numpy as np
from collections import Counter

# Load API key from .env
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Initialize API
api = wandb.Api()

# Get all runs from the project
runs = api.runs("unreflect-anything/UnReflectAnything")

print(f"Total runs: {len(runs)}")
print("Computing total energy consumption...\n")

# Exact metric names to look for (based on WandB system metrics)
EXACT_POWER_METRICS = [
    "Process GPU 0 Power Usage (W)",
    "Process GPU 1 Power Usage (W)",
    "Process GPU 2 Power Usage (W)",
    "Process GPU 3 Power Usage (W)",
    "system/process/gpu.0.powerWatts",
    "system/process/gpu.1.powerWatts",
    "system/process/gpu.2.powerWatts",
    "system/process/gpu.3.powerWatts",
]

# Priority order: most specific first
POWER_METRIC_PRIORITY = [
    'process gpu',  # Matches "Process GPU 0 Power Usage (W)" and similar
    'system/gpu.*power',
    'system/gpu.*energy',
    'gpu.*power',
    'gpu.*energy',
    'system/cpu.*power',
    'system/cpu.*energy',
    'power',
    'energy',
]

def find_power_metrics(history_keys):
    """
    Find power-related metrics in history keys.
    Prioritizes exact GPU power metrics, especially "Process GPU X Power Usage (W)".
    """
    power_metrics = []
    priority_metrics = []
    exact_matches = []
    
    # First, check for exact matches (highest priority)
    for exact_metric in EXACT_POWER_METRICS:
        if exact_metric in history_keys:
            exact_matches.append(exact_metric)
    
    # Then check for pattern matches
    for key in history_keys:
        key_lower = key.lower()
        
        # Check for power or energy keywords
        if 'power' in key_lower or 'energy' in key_lower:
            # Prioritize "Process GPU" metrics
            if 'process gpu' in key_lower and 'power' in key_lower:
                if key not in exact_matches:  # Don't duplicate
                    priority_metrics.append(key)
            # Then check other patterns
            elif any(pattern.replace('.*', '') in key_lower for pattern in POWER_METRIC_PRIORITY):
                if key not in exact_matches and key not in priority_metrics:  # Don't duplicate
                    power_metrics.append(key)
    
    # Return exact matches first, then priority metrics, then others
    return exact_matches + priority_metrics + power_metrics

def integrate_power_over_time(history, power_metric, time_metric='_timestamp'):
    """
    Integrate power over time to get energy.
    
    Args:
        history: List of dicts with metric values
        power_metric: Key name for power metric
        time_metric: Key name for time metric (default: '_timestamp')
    
    Returns:
        Energy in Watt-hours (Wh)
    """
    if len(history) < 2:
        return 0.0
    
    # Extract power and time values
    powers = []
    times = []
    
    for entry in history:
        try:
            if power_metric in entry and entry[power_metric] is not None:
                power_val = float(entry[power_metric])
                # Get timestamp
                if time_metric in entry and entry[time_metric] is not None:
                    time_val = float(entry[time_metric])
                else:
                    # Fallback: use step index as time proxy
                    time_val = len(times)
                
                powers.append(power_val)
                times.append(time_val)
        except (ValueError, TypeError, KeyError):
            continue
    
    if len(powers) < 2:
        return 0.0
    
    # Convert to numpy arrays for efficient computation
    powers = np.array(powers)  # Shape: [N]
    times = np.array(times)  # Shape: [N]
    
    # Sort by time to ensure chronological order
    sort_idx = np.argsort(times)
    powers = powers[sort_idx]
    times = times[sort_idx]
    
    # Convert timestamps to seconds (assuming Unix timestamps)
    # If times are already in seconds or are step indices, use as-is
    if times[0] > 1e10:  # Likely Unix timestamp in milliseconds
        times = times / 1000.0
    elif times[0] > 1e9:  # Likely Unix timestamp in seconds
        pass  # Already in seconds
    # Otherwise assume it's relative time or step-based
    
    # Compute time differences
    time_diffs = np.diff(times)  # Shape: [N-1]
    
    # Use trapezoidal rule for integration: ∫P(t)dt ≈ Σ(P_i + P_{i+1})/2 * Δt_i
    # Energy = ∫ Power dt (in Joules if power in Watts and time in seconds)
    # Convert to Watt-hours: 1 Wh = 3600 J
    power_avg = (powers[:-1] + powers[1:]) / 2.0  # Shape: [N-1]
    energy_joules = np.sum(power_avg * time_diffs)  # Shape: scalar
    energy_wh = energy_joules / 3600.0  # Convert to Watt-hours
    
    return energy_wh

# Statistics
total_energy_wh = 0.0
runs_with_energy = 0
runs_processed = 0
runs_failed = 0
runs_no_data = 0

run_energies = []

for i, run in enumerate(runs):
    try:
        runs_processed += 1
        
        # Safely get run info
        run_name = getattr(run, 'name', f'run_{i+1}')
        run_id = getattr(run, 'id', 'Unknown')
        run_state = getattr(run, 'state', 'Unknown')
        
        # Try to get history - use stream="events" for system metrics
        history = None
        try:
            # First try to get system metrics (events stream)
            history = run.history(stream="events", pandas=False)
        except Exception:
            # Fallback to regular history
            try:
                history = run.history(pandas=False)
            except Exception:
                runs_no_data += 1
                print(f"[{runs_processed:4d}/{len(runs)}] {run_name[:40]:<40} | State: {run_state:<10} | ❌ No history")
                continue
        
        if history is None or len(history) == 0:
            runs_no_data += 1
            print(f"[{runs_processed:4d}/{len(runs)}] {run_name[:40]:<40} | State: {run_state:<10} | ⚠️  Empty history")
            continue
        
        # Get available keys
        try:
            history_keys = list(history[0].keys())
        except (KeyError, IndexError, AttributeError):
            runs_no_data += 1
            print(f"[{runs_processed:4d}/{len(runs)}] {run_name[:40]:<40} | State: {run_state:<10} | ❌ Cannot read keys")
            continue
        
        # Debug: show all keys for first few runs to identify metric names
        if runs_processed <= 3:
            print(f"\n  DEBUG - Run {runs_processed} ({run_name}):")
            print(f"    Total history entries: {len(history)}")
            print(f"    Available keys ({len(history_keys)}):")
            for key in sorted(history_keys):
                # Show sample value if available
                try:
                    sample_val = history[0].get(key, 'N/A')
                    if isinstance(sample_val, (int, float)):
                        print(f"      - {key}: {sample_val}")
                    else:
                        print(f"      - {key}: {type(sample_val).__name__}")
                except Exception:
                    print(f"      - {key}")
            print()
        
        # Find power metrics
        power_metrics = find_power_metrics(history_keys)
        
        if not power_metrics:
            runs_no_data += 1
            print(f"[{runs_processed:4d}/{len(runs)}] {run_name[:40]:<40} | State: {run_state:<10} | ⚠️  No power metrics ({len(history)} entries)")
            if runs_processed <= 3:
                print(f"    Available keys (first 20): {sorted(history_keys)[:20]}")
            continue
        
        # Try each power metric and use the first one that works
        # If multiple GPUs are found, sum their energy
        run_energy_wh = 0.0
        metrics_used = []
        power_stats = None
        all_powers = []
        
        for power_metric in power_metrics:
            try:
                # Get power values for statistics
                powers = []
                for entry in history:
                    try:
                        if power_metric in entry and entry[power_metric] is not None:
                            power_val = float(entry[power_metric])
                            powers.append(power_val)
                            all_powers.append(power_val)
                    except (ValueError, TypeError, KeyError):
                        continue
                
                if len(powers) > 0:
                    # Compute energy for this metric
                    energy = integrate_power_over_time(history, power_metric)
                    if energy > 0:
                        run_energy_wh += energy
                        metrics_used.append(power_metric)
                        
                        # Update power stats (aggregate across all GPUs)
                        if power_stats is None:
                            power_stats = {
                                'min': float(np.min(powers)),
                                'max': float(np.max(powers)),
                                'mean': float(np.mean(powers)),
                                'count': len(powers)
                            }
                        else:
                            power_stats['min'] = min(power_stats['min'], float(np.min(powers)))
                            power_stats['max'] = max(power_stats['max'], float(np.max(powers)))
                            # Weighted mean
                            total_count = power_stats['count'] + len(powers)
                            power_stats['mean'] = (power_stats['mean'] * power_stats['count'] + np.mean(powers) * len(powers)) / total_count
                            power_stats['count'] = total_count
            except Exception as e:
                if runs_processed <= 3:
                    print(f"    Error processing metric {power_metric}: {str(e)}")
                continue
        
        # If we have aggregated power stats from all metrics, use those
        if all_powers and power_stats is None:
            power_stats = {
                'min': float(np.min(all_powers)),
                'max': float(np.max(all_powers)),
                'mean': float(np.mean(all_powers)),
                'count': len(all_powers)
            }
        
        metric_used = ", ".join(metrics_used) if metrics_used else None
        
        if run_energy_wh > 0:
            total_energy_wh += run_energy_wh
            runs_with_energy += 1
            run_energies.append({
                'name': run_name,
                'id': run_id,
                'state': run_state,
                'energy_wh': run_energy_wh,
                'metric': metric_used
            })
            
            # Print run data
            if power_stats:
                print(f"[{runs_processed:4d}/{len(runs)}] {run_name[:40]:<40} | State: {run_state:<10} | "
                      f"✅ Energy: {run_energy_wh:8.2f} Wh ({run_energy_wh/1000:6.4f} kWh) | "
                      f"Power: {power_stats['mean']:6.1f}W (min: {power_stats['min']:5.1f}W, max: {power_stats['max']:5.1f}W) | "
                      f"Samples: {power_stats['count']:5d} | Metric: {metric_used[:30]}")
            else:
                print(f"[{runs_processed:4d}/{len(runs)}] {run_name[:40]:<40} | State: {run_state:<10} | "
                      f"✅ Energy: {run_energy_wh:8.2f} Wh ({run_energy_wh/1000:6.4f} kWh) | "
                      f"Metric: {metric_used[:50]}")
        else:
            runs_no_data += 1
            print(f"[{runs_processed:4d}/{len(runs)}] {run_name[:40]:<40} | State: {run_state:<10} | "
                  f"⚠️  Could not compute energy ({len(history)} entries, {len(power_metrics)} power metrics)")
        
    except Exception as e:
        runs_failed += 1
        if runs_failed <= 5:  # Only print first few errors
            print(f"Error processing run {i+1}: {str(e)}")
        continue

# Print comprehensive report
print(f"\n{'='*80}")
print("ENERGY CONSUMPTION REPORT")
print(f"{'='*80}\n")

print(f"Total runs processed: {runs_processed}")
print(f"Runs with energy data: {runs_with_energy}")
print(f"Runs without data: {runs_no_data}")
print(f"Runs failed: {runs_failed}\n")

if total_energy_wh > 0:
    # Convert to different units
    total_energy_kwh = total_energy_wh / 1000.0
    total_energy_mwh = total_energy_kwh / 1000.0
    
    # CO2 equivalent (assuming average grid mix: ~0.5 kg CO2/kWh)
    co2_kg = total_energy_kwh * 0.5
    
    print(f"{'='*80}")
    print("TOTAL ENERGY CONSUMPTION")
    print(f"{'='*80}")
    print(f"Total Energy: {total_energy_wh:.4f} Wh")
    print(f"Total Energy: {total_energy_kwh:.4f} kWh")
    print(f"Total Energy: {total_energy_mwh:.6f} MWh")
    print(f"\nEstimated CO2 Equivalent: {co2_kg:.4f} kg CO2")
    print(f"Estimated CO2 Equivalent: {co2_kg/1000:.6f} metric tons CO2")
    
    if runs_with_energy > 0:
        avg_energy_per_run = total_energy_wh / runs_with_energy
        print(f"\nAverage Energy per Run: {avg_energy_per_run:.4f} Wh")
        print(f"Average Energy per Run: {avg_energy_per_run/1000:.4f} kWh")
    
    # Show metric usage statistics
    metric_counter = Counter([r['metric'] for r in run_energies if r['metric']])
    if metric_counter:
        print(f"\n{'='*80}")
        print("METRICS USED")
        print(f"{'='*80}")
        print(f"{'Metric Name':<50} {'Runs Using It':<15}")
        print("-" * 80)
        for metric, count in metric_counter.most_common():
            print(f"{metric[:48]:<50} {count:<15}")
    
    # Show top energy-consuming runs
    if run_energies:
        print(f"\n{'='*80}")
        print("TOP 10 ENERGY-CONSUMING RUNS")
        print(f"{'='*80}")
        sorted_runs = sorted(run_energies, key=lambda x: x['energy_wh'], reverse=True)
        print(f"{'Rank':<6} {'Run Name':<30} {'Energy (Wh)':<15} {'Energy (kWh)':<15} {'State':<10}")
        print("-" * 80)
        for idx, run_info in enumerate(sorted_runs[:10], 1):
            print(f"{idx:<6} {run_info['name'][:28]:<30} {run_info['energy_wh']:<15.4f} {run_info['energy_wh']/1000:<15.4f} {run_info['state']:<10}")
    
    print(f"\n{'='*80}")
    print("Report complete!")
else:
    print("No energy data found in any runs.")
    print("\nTroubleshooting:")
    print("1. Check if power/energy metrics are being logged to WandB")
    print("2. Verify metric names match expected patterns")
    print("3. Ensure runs have history data available")
