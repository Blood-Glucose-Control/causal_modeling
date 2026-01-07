import warnings
# Suppress scipy stats warnings about small sample sizes
warnings.filterwarnings("ignore", message=".*p-value may be inaccurate with fewer than 20 observations.*")

from its_package.pipeline.analysis_pipeline import CausalAnalysisPipeline

def test_single_analysis():
    """Test function for the single dataset analysis"""
    # Initialize pipeline
    pipeline = CausalAnalysisPipeline(output_dir="test_output")
    
    # Set data path (replace with your actual data path)
    data_path = "../synthetic_data/data/ml_dataset.csv"
    
    print(f"Running single dataset analysis with data from: {data_path}")
    
    # Run analysis with fewer events for quick testing
    results = pipeline.run_single_dataset_analysis(
        data_path=data_path,
        max_events=2,
        pre_window='45min',
        post_window='30min'
    )
    
    # Save results
    pipeline.save_results("test_single_analysis_results.json")
    
    print("Analysis complete!")
    return results

def test_counterfactual_analysis():
    """Test function for the counterfactual analysis"""
    # Initialize pipeline
    pipeline = CausalAnalysisPipeline(output_dir="test_output")
    
    # Set data path (replace with your actual counterfactual data path)
    data_path = "../synthetic_data/data/dose_counterfactuals"
    
    print(f"Running counterfactual analysis with data from: {data_path}")
    
    # Run analysis with fewer events for quick testing
    results = pipeline.run_counterfactual_analysis(
        data_path=data_path,
        max_events=2,
        pre_window='45min',
        post_window='30min'
    )
    
    # Save results
    pipeline.save_results("test_counterfactual_analysis_results.json")
    
    print("Analysis complete!")
    return results

if __name__ == "__main__":
    print("=" * 50)
    print("ITS PACKAGE TEST SCRIPT")
    print("=" * 50)
    
    # Choose which test to run
    test_type = input("Run which test? (single/counterfactual/both) [single]: ") or "single"
    
    if test_type.lower() in ['single', 'both']:
        print("\nRunning single dataset analysis test...")
        test_single_analysis()
    
    if test_type.lower() in ['counterfactual', 'both']:
        print("\nRunning counterfactual analysis test...")
        test_counterfactual_analysis()
    
    print("\nAll tests completed!")
