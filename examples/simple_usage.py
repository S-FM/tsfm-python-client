"""
Simple usage examples for TSFM Client - No async/await needed!
"""

import pandas as pd
import numpy as np

# Import the simple interface
from tsfm_client import SimpleTSFMClient, predict


def main():
    """
    Examples using the simple synchronous interface
    """
    print("🚀 TSFM Client - Simple Usage Examples")
    print("=" * 50)
    
    # Method 1: Quick one-off prediction
    print("\n1. Quick Prediction (Global Function)")
    print("-" * 40)
    
    data = [10, 12, 13, 15, 17, 16, 18, 20, 22, 25]
    print(f"Input data: {data}")
    
    try:
        response = predict(
            data=data,
            forecast_horizon=5
        )
        print(f"✅ Forecast: {response.forecast}")
        print(f"   Model: {response.model_name}")
        print(f"   Metadata: {response.metadata}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Method 2: Client instance for multiple predictions
    print("\n2. Client Instance (Multiple Predictions)")
    print("-" * 40)
    
    client = SimpleTSFMClient()  # Uses TSFM_API_KEY environment variable
    
    try:
        # List available models
        models = client.list_models()
        print(f"Available models: {models}")
        
        # Simple prediction
        response = client.predict([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        print(f"✅ Simple forecast: {response.forecast}")
        
        # With confidence intervals
        response = client.predict(
            data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            confidence_intervals=True,
            forecast_horizon=3
        )
        print(f"✅ Forecast with CI: {response.forecast}")
        if response.confidence_intervals:
            print(f"   Confidence intervals: {response.confidence_intervals}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Method 3: With pandas Series
    print("\n3. Using Pandas Series")
    print("-" * 40)
    
    try:
        # Create sample time series
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        ts_data = pd.Series(np.random.randn(20).cumsum() + 100, index=dates)
        
        print(f"Pandas series shape: {ts_data.shape}")
        print(f"Sample values: {ts_data.head().tolist()}")
        
        response = client.predict(
            data=ts_data,
            forecast_horizon=7,
            model="chronos-t5-small"
        )
        print(f"✅ Forecast: {response.forecast}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Method 4: Different models
    print("\n4. Different Models")
    print("-" * 40)
    
    test_data = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    try:
        # Chronos T5 Small
        response1 = client.predict(
            data=test_data,
            model="chronos-t5-small",
            forecast_horizon=3
        )
        print(f"✅ Chronos T5 Small: {response1.forecast}")
        
        # TOTO Open Base (if available)
        try:
            response2 = client.predict(
                data=test_data,
                model="toto-open-base-1.0",
                forecast_horizon=3
            )
            print(f"✅ TOTO Open Base: {response2.forecast}")
        except Exception as e:
            print(f"⚠️  TOTO model not available: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Method 5: Model info and health check
    print("\n5. Model Info & Health Check")
    print("-" * 40)
    
    try:
        # Health check
        health = client.health_check()
        print(f"✅ Health check: {health}")
        
        # Model info
        info = client.get_model_info("chronos-t5-small")
        print(f"✅ Model info: {info}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 All examples completed!")


def pandas_example():
    """
    Advanced pandas integration example
    """
    print("\n📊 Advanced Pandas Example")
    print("=" * 30)
    
    # Create realistic time series data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    
    # Simulate sales data with trend and noise
    trend = np.linspace(100, 200, 50)
    noise = np.random.normal(0, 10, 50)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(50) / 7)  # Weekly seasonality
    
    sales_data = pd.Series(trend + seasonal + noise, index=dates, name='sales')
    
    print(f"Sample data (last 5 days):")
    print(sales_data.tail())
    
    try:
        # Make prediction
        response = predict(
            data=sales_data,
            forecast_horizon=10,
            confidence_intervals=True
        )
        
        print(f"\n✅ 10-day forecast: {response.forecast}")
        
        # Convert to DataFrame for easier analysis
        forecast_df = response.to_pandas()
        print(f"\nForecast DataFrame:")
        print(forecast_df)
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("Make sure the TSFM server is running on http://localhost:8000")
    print()
    
    # Run main examples
    main()
    
    # Run pandas example
    pandas_example()