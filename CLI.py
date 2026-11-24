# CLI.py
# -*- coding: utf-8 -*-
"""
Device Test Data Analyzer CLI
Usage: python CLI.py data_file.csv [OPTIONS]
"""

import sys
import os
import click
import pandas as pd
import traceback
import json

# Add current directory to path for module imports
current_path = os.getcwd()
sys.path.insert(0, current_path)

# Try to import required modules
try:
    from Analyzer import DeviceTestAnalyzer
    from DataLoader import load_and_validate_data
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

@click.command()
@click.argument('data_file', type=click.Path(exists=True, readable=True))
@click.option('--output', '-o', default='analysis_results', 
              help='Output directory for results (default: analysis_results)')
@click.option('--full/--quick', 'analysis_mode', default=True,
              help='Full analysis with visualizations or quick summary (default: full)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def main(data_file, output, analysis_mode, verbose):
    """
    Analyze device test data from CSV files.
    
    DATA_FILE: Path to CSV file containing test results
    """
    
    # Check imports first
    if not IMPORT_SUCCESS:
        click.echo("ERROR: Failed to import required modules")
        click.echo("Details: {}".format(IMPORT_ERROR))
        click.echo("Current path: {}".format(current_path))
        click.echo("Python version: {}".format(sys.version))
        click.echo("sys.path: {}".format(sys.path[:3]))
        sys.exit(1)
    
    # Set verbosity
    if verbose:
        click.echo("VERBOSE: Verbose mode enabled")
        click.echo("        Script location: {}".format(__file__))
        click.echo("        Working directory: {}".format(os.getcwd()))
        click.echo("        Python executable: {}".format(sys.executable))
        click.echo("        Python version: {}".format(sys.version))
    
    click.echo("\n" + "=" * 60)
    click.echo("DEVICE TEST DATA ANALYZER")
    click.echo("=" * 60)
    
    try:
        # 1. Create output directory
        os.makedirs(output, exist_ok=True)
        if verbose:
            click.echo("OUTPUT: Output directory: {}".format(os.path.abspath(output)))
        
        # 2. Load and validate data
        click.echo("\nLOADING DATA")
        click.echo("   Source: {}".format(data_file))
        
        try:
            df = load_and_validate_data(data_file)
        except Exception as e:
            click.echo("ERROR: Data loading failed: {}".format(str(e)))
            sys.exit(1)
        
        # Validate DataFrame
        if df is None:
            click.echo("ERROR: No data returned from loader")
            sys.exit(1)
        
        if not isinstance(df, pd.DataFrame):
            click.echo("ERROR: Expected DataFrame, got {}".format(type(df)))
            sys.exit(1)
        
        if df.empty:
            click.echo("ERROR: Loaded DataFrame is empty")
            sys.exit(1)
        
        click.echo("SUCCESS: Data loaded successfully")
        click.echo("        - Records: {:,}".format(len(df)))
        click.echo("        - Columns: {}".format(df.shape[1]))
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        click.echo("        - Memory: {:.2f} MB".format(memory_mb))
        
        if verbose:
            click.echo("\nCOLUMNS:")
            for i, col in enumerate(df.columns, 1):
                dtype = df[col].dtype
                click.echo("   {:2d}. {} ({})".format(i, col, dtype))
        
        # 3. Create analyzer
        click.echo("\nCREATING ANALYZER")
        try:
            analyzer = DeviceTestAnalyzer(df)
            click.echo("SUCCESS: Analyzer initialized")
        except Exception as e:
            click.echo("ERROR: Analyzer creation failed: {}".format(str(e)))
            if verbose:
                traceback.print_exc()
            sys.exit(1)
        
        # 4. Run analysis based on mode
        if analysis_mode:  # Full analysis
            click.echo("\nRUNNING FULL ANALYSIS")
            click.echo("   This includes:")
            click.echo("   - Statistical summaries")
            click.echo("   - Hypothesis testing")
            click.echo("   - Data visualizations")
            click.echo("   - Detailed reports")
            
            try:
                analyzer.run_full_analysis(output)
                click.echo("\nSUCCESS: FULL ANALYSIS COMPLETE")
            except Exception as e:
                click.echo("\nERROR: Full analysis failed: {}".format(str(e)))
                if verbose:
                    traceback.print_exc()
                sys.exit(1)
                
        else:  # Quick analysis
            click.echo("\nRUNNING QUICK ANALYSIS")
            
            try:
                results = analyzer.exploratory_analysis()
                
                click.echo("\nQUICK ANALYSIS RESULTS")
                click.echo("-" * 40)
                
                # Display key metrics
                if isinstance(results, dict):
                    for key, value in results.items():
                        if key != 'data_quality':
                            # Format based on value type
                            if isinstance(value, (int, float)):
                                key_display = key.replace('_', ' ').title()
                                if 'rate' in key.lower() or 'percent' in key.lower():
                                    click.echo("   {}: {:.2%}".format(key_display, value))
                                elif 'avg' in key.lower() or 'mean' in key.lower():
                                    click.echo("   {}: {:.2f}".format(key_display, value))
                                else:
                                    click.echo("   {}: {:,}".format(key_display, value))
                            else:
                                click.echo("   {}: {}".format(key.replace('_', ' ').title(), value))
                else:
                    click.echo("   Results: {}".format(results))
                
                # Try hypothesis test if available
                try:
                    hypothesis_results = analyzer.hypothesis_test()
                    if hypothesis_results and 'z_test' in hypothesis_results:
                        click.echo("\nHYPOTHESIS TESTING")
                        z_result = hypothesis_results['z_test']
                        p_value = z_result.get('p_value', 'N/A')
                        if isinstance(p_value, (int, float)):
                            significance = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
                            click.echo("   - Z-test p-value: {:.6f} ({})".format(p_value, significance))
                except Exception as e:
                    if verbose:
                        click.echo("   WARNING: Hypothesis test skipped: {}".format(str(e)))
                
            except Exception as e:
                click.echo("\nERROR: Quick analysis failed: {}".format(str(e)))
                if verbose:
                    traceback.print_exc()
                sys.exit(1)
        
        # 5. Show completion message
        click.echo("\n" + "=" * 60)
        click.echo("ANALYSIS COMPLETED SUCCESSFULLY!")
        click.echo("Results saved to: {}".format(os.path.abspath(output)))
        click.echo("=" * 60)
        
        # List generated files
        if os.path.exists(output) and os.path.isdir(output):
            files = [f for f in os.listdir(output) if os.path.isfile(os.path.join(output, f))]
            if files:
                click.echo("\nGENERATED FILES:")
                for file in sorted(files):
                    size = os.path.getsize(os.path.join(output, file))
                    click.echo("   - {} ({:,} bytes)".format(file, size))
        
    except KeyboardInterrupt:
        click.echo("\n\nWARNING: Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        click.echo("\nERROR: UNEXPECTED ERROR: {}".format(str(e)))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
