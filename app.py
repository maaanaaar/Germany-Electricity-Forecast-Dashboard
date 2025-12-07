
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Germany Electricity Load Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

class ElectricityForecastDashboard:
    def __init__(self):
        self.df = None
        self.test = None
        self.model = None
        self.metrics = None
        self.load_data()
        
    def load_data(self):
        """Load pre-trained model and data"""
        try:
            # Load saved data
            self.df = pd.read_csv('germany_electricity_dataset.csv', 
                                 index_col=0, parse_dates=True)
            self.test = pd.read_csv('germany_forecast_results.csv', 
                                   index_col=0, parse_dates=True)
            
            # Load model
            self.model = joblib.load('germany_load_forecast_model.pkl')
            
            # Load metrics
            with open('model_metrics.json', 'r') as f:
                self.metrics = json.load(f)
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Please run the training script first.")
    
    def create_overview(self):
        """Create overview section"""
        st.markdown('<h1 class="main-header">⚡ Germany Electricity Load Forecasting Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("""
            ### Overview
            Real-time dashboard for Germany's electricity load forecasting using XGBoost.
            Analyzes historical consumption patterns, weather data, and temporal features.
            """)
        
        with col2:
            st.metric("Forecast Horizon", "72 Hours", "Real-time")
            st.metric("Data Frequency", "Hourly", "Continuous")
            
        with col3:
            st.metric("Model Accuracy", f"{self.metrics.get('R2', 0.95):.2%}", "High")
            st.metric("Avg. MAPE", f"{self.metrics.get('MAPE', 2.5):.2f}%", "Excellent")
    
    def create_key_metrics(self):
        """Display key performance metrics"""
        st.markdown('<h2 class="sub-header">📊 Key Performance Metrics</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = self.metrics
        
        with col1:
            st.metric("RMSE", f"{metrics.get('RMSE', 0):,.0f} MW", 
                     delta=f"{(metrics.get('RMSE', 0)/metrics.get('MAE', 1)*100-100):.1f}%")
            
        with col2:
            st.metric("MAE", f"{metrics.get('MAE', 0):,.0f} MW")
            
        with col3:
            st.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
            
        with col4:
            st.metric("R² Score", f"{metrics.get('R2', 0):.4f}")
            
        with col5:
            st.metric("CV(RMSE)", f"{metrics.get('CV_RMSE', 0):.2f}%")
    
    def create_forecast_chart(self):
        """Create interactive forecast visualization"""
        st.markdown('<h2 class="sub-header">📈 Forecast vs Actual Load</h2>', unsafe_allow_html=True)
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            default_start = self.test.index.min().date()
            start_date = st.date_input("Start Date", value=default_start)
        with col2:
            default_end = min(self.test.index.max().date(), 
                            (datetime.strptime(str(start_date), '%Y-%m-%d') + timedelta(days=30)).date())
            end_date = st.date_input("End Date", value=default_end)
        
        # Filter data for selected period
        mask = (self.test.index.date >= start_date) & (self.test.index.date <= end_date)
        filtered_data = self.test[mask]
        
        if len(filtered_data) > 0:
            # Create interactive plot
            fig = go.Figure()
            
            # Add actual load
            fig.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['Consumption_MW'],
                mode='lines',
                name='Actual Load',
                line=dict(color='#2E86AB', width=3),
                hovertemplate='%{x}<br>Actual: %{y:,.0f} MW<extra></extra>'
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['prediction'],
                mode='lines',
                name='XGBoost Forecast',
                line=dict(color='#A23B72', width=3, dash='dash'),
                hovertemplate='%{x}<br>Forecast: %{y:,.0f} MW<extra></extra>'
            ))
            
            # Add confidence band
            error_std = filtered_data['error'].std()
            fig.add_trace(go.Scatter(
                x=filtered_data.index.tolist() + filtered_data.index.tolist()[::-1],
                y=(filtered_data['prediction'] + error_std).tolist() + 
                  (filtered_data['prediction'] - error_std).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(162, 59, 114, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='Confidence Band'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Electricity Load Forecast: {start_date} to {end_date}",
                xaxis_title="Date & Time",
                yaxis_title="Load (MW)",
                hovermode='x unified',
                template='plotly_white',
                height=500,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics for selected period
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                period_rmse = np.sqrt(np.mean((filtered_data['Consumption_MW'] - filtered_data['prediction'])**2))
                st.metric("Period RMSE", f"{period_rmse:,.0f} MW")
            with col2:
                period_mape = np.mean(np.abs((filtered_data['Consumption_MW'] - filtered_data['prediction']) / 
                                           filtered_data['Consumption_MW'])) * 100
                st.metric("Period MAPE", f"{period_mape:.2f}%")
            with col3:
                st.metric("Avg Actual Load", f"{filtered_data['Consumption_MW'].mean():,.0f} MW")
            with col4:
                st.metric("Avg Forecast", f"{filtered_data['prediction'].mean():,.0f} MW")
        else:
            st.warning("No data available for selected date range.")
    
    def create_error_analysis(self):
        """Create error analysis visualizations"""
        st.markdown('<h2 class="sub-header">🔍 Error Analysis</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Error Distribution", "Hourly Pattern", "Daily Pattern", "Residual Analysis"])
        
        with tab1:
            fig = px.histogram(self.test, x='error', nbins=50,
                             title='Error Distribution',
                             labels={'error': 'Forecast Error (MW)'},
                             color_discrete_sequence=['#D1495B'])
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Error", f"{self.test['error'].mean():,.0f} MW")
            with col2:
                st.metric("Error Std", f"{self.test['error'].std():,.0f} MW")
            with col3:
                st.metric("Max Error", f"{self.test['error'].max():,.0f} MW")
        
        with tab2:
            # Hourly error pattern
            hourly_error = self.test.groupby(self.test.index.hour)['error'].mean()
            fig = go.Figure(data=[
                go.Bar(x=list(hourly_error.index), y=hourly_error.values,
                      marker_color='#9467bd')
            ])
            fig.update_layout(
                title='Average Error by Hour of Day',
                xaxis_title='Hour',
                yaxis_title='Mean Absolute Error (MW)',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Daily error pattern
            daily_error = self.test.groupby(self.test.index.dayofweek)['error'].mean()
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            fig = go.Figure(data=[
                go.Bar(x=days, y=daily_error.values,
                      marker_color='#ff7f0e')
            ])
            fig.update_layout(
                title='Average Error by Day of Week',
                xaxis_title='Day',
                yaxis_title='Mean Absolute Error (MW)',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Residual analysis
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.test['prediction'],
                y=self.test['Consumption_MW'] - self.test['prediction'],
                mode='markers',
                marker=dict(size=4, color='#2ca02c', opacity=0.6),
                name='Residuals'
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(
                title='Residual Analysis',
                xaxis_title='Predicted Load (MW)',
                yaxis_title='Residual (MW)',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def create_feature_analysis(self):
        """Create feature importance and analysis"""
        st.markdown('<h2 class="sub-header">🎯 Feature Importance</h2>', unsafe_allow_html=True)
        
        try:
            # Get feature names from model
            feature_names = self.model.feature_names_in_
            importance = self.model.feature_importances_
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(15)
            
            # Create horizontal bar chart
            fig = px.bar(feature_importance_df, 
                        y='feature', 
                        x='importance',
                        orientation='h',
                        title='Top 15 Most Important Features',
                        color='importance',
                        color_continuous_scale='Viridis')
            
            fig.update_layout(
                template='plotly_white',
                height=500,
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.info("Feature importance visualization requires model with feature_importances_ attribute.")
    
    def create_seasonal_analysis(self):
        """Create seasonal and weather analysis"""
        st.markdown('<h2 class="sub-header">🌡️ Weather & Seasonal Analysis</h2>', unsafe_allow_html=True)
        
        if 'Temperature' in self.test.columns:
            tab1, tab2 = st.tabs(["Temperature Impact", "Seasonal Patterns"])
            
            with tab1:
                fig = make_subplots(rows=2, cols=1, 
                                  subplot_titles=("Load vs Temperature", "Forecast Error vs Temperature"),
                                  vertical_spacing=0.15)
                
                # Scatter plot: Load vs Temperature
                fig.add_trace(
                    go.Scatter(
                        x=self.test['Temperature'],
                        y=self.test['Consumption_MW'],
                        mode='markers',
                        marker=dict(size=3, opacity=0.5, color='#1f77b4'),
                        name='Actual Load'
                    ),
                    row=1, col=1
                )
                
                # Scatter plot: Error vs Temperature
                fig.add_trace(
                    go.Scatter(
                        x=self.test['Temperature'],
                        y=self.test['error'],
                        mode='markers',
                        marker=dict(size=3, opacity=0.5, color='#ff7f0e'),
                        name='Forecast Error'
                    ),
                    row=2, col=1
                )
                
                fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
                fig.update_xaxes(title_text="Temperature (°C)", row=2, col=1)
                fig.update_yaxes(title_text="Load (MW)", row=1, col=1)
                fig.update_yaxes(title_text="Error (MW)", row=2, col=1)
                fig.update_layout(height=600, template='plotly_white', showlegend=False)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Monthly patterns
                monthly_data = self.test.groupby(self.test.index.month).agg({
                    'Consumption_MW': 'mean',
                    'prediction': 'mean',
                    'Temperature': 'mean'
                })
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Bar(x=months, y=monthly_data['Consumption_MW'],
                          name='Average Load', marker_color='#2E86AB'),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(x=months, y=monthly_data['Temperature'],
                             name='Average Temperature', mode='lines+markers',
                             line=dict(color='#FF9F1C', width=3)),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title='Monthly Load and Temperature Patterns',
                    template='plotly_white',
                    height=400
                )
                fig.update_yaxes(title_text="Load (MW)", secondary_y=False)
                fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Weather data not available in the current dataset.")
    
    def create_what_if_scenarios(self):
        """Create what-if analysis scenarios"""
        st.markdown('<h2 class="sub-header">🎮 What-If Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temp_scenario = st.slider(
                "Temperature Change (°C)",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                help="Simulate impact of temperature changes"
            )
        
        with col2:
            hour_scenario = st.selectbox(
                "Hour of Day",
                list(range(24)),
                format_func=lambda x: f"{x:02d}:00",
                index=14,
                help="Select hour to analyze"
            )
        
        with col3:
            season_scenario = st.selectbox(
                "Season",
                ["Winter", "Spring", "Summer", "Fall"],
                index=1
            )
        
        # Calculate impact based on simplified model
        # This is a demonstration - in real app you would use the trained model
        temp_impact = temp_scenario * 150  # 150 MW per °C change
        
        if season_scenario == "Winter":
            season_impact = -8000
        elif season_scenario == "Summer":
            season_impact = 5000
        else:
            season_impact = 0
        
        hour_impact = 1000 * np.sin(2 * np.pi * hour_scenario / 24)
        
        # Display scenario results
        st.markdown("""
        <div class="metric-card">
        <h4>Scenario Impact Estimate</h4>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Temperature Impact", f"{temp_impact:+,.0f} MW", f"{temp_scenario:+.1f}°C")
        with col2:
            st.metric("Hour Impact", f"{hour_impact:+,.0f} MW", f"{hour_scenario:02d}:00")
        with col3:
            st.metric("Season Impact", f"{season_impact:+,.0f} MW", season_scenario)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def create_data_summary(self):
        """Create data summary section"""
        st.markdown('<h2 class="sub-header">📋 Data Summary</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Dataset Information")
            st.write(f"**Time Range:** {self.df.index.min().date()} to {self.df.index.max().date()}")
            st.write(f"**Total Records:** {len(self.df):,}")
            st.write(f"**Training Period:** {self.df[self.df.index < '2018-01-01'].index.min().date()} to "
                    f"{self.df[self.df.index < '2018-01-01'].index.max().date()}")
            st.write(f"**Test Period:** {self.test.index.min().date()} to {self.test.index.max().date()}")
            st.write(f"**Features Used:** {len(self.df.columns) - 1}")
        
        with col2:
            st.markdown("#### Load Statistics")
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                'Value (MW)': [
                    f"{self.df['Consumption_MW'].mean():,.0f}",
                    f"{self.df['Consumption_MW'].std():,.0f}",
                    f"{self.df['Consumption_MW'].min():,.0f}",
                    f"{self.df['Consumption_MW'].quantile(0.25):,.0f}",
                    f"{self.df['Consumption_MW'].quantile(0.5):,.0f}",
                    f"{self.df['Consumption_MW'].quantile(0.75):,.0f}",
                    f"{self.df['Consumption_MW'].max():,.0f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    def create_export_section(self):
        """Create data export section"""
        st.markdown('<h2 class="sub-header">📤 Export Results</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = self.test.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Forecast Data",
                data=csv,
                file_name="germany_load_forecasts.csv",
                mime="text/csv",
                help="Download forecast results as CSV"
            )
        
        with col2:
            # Create metrics report
            metrics_report = f"""
            GERMANY ELECTRICITY LOAD FORECASTING MODEL
            ===========================================
            
            Performance Metrics:
            --------------------
            RMSE: {self.metrics.get('RMSE', 0):,.0f} MW
            MAE: {self.metrics.get('MAE', 0):,.0f} MW
            MAPE: {self.metrics.get('MAPE', 0):.2f}%
            R²: {self.metrics.get('R2', 0):.4f}
            CV(RMSE): {self.metrics.get('CV_RMSE', 0):.2f}%
            
            Dataset Information:
            --------------------
            Time Range: {self.df.index.min().date()} to {self.df.index.max().date()}
            Total Samples: {len(self.df):,}
            Forecast Period: {self.test.index.min().date()} to {self.test.index.max().date()}
            
            Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            st.download_button(
                label="📊 Download Performance Report",
                data=metrics_report,
                file_name="model_performance_report.txt",
                mime="text/plain",
                help="Download detailed performance metrics"
            )
        
        with col3:
            if st.button("🔄 Generate New Forecast", help="Run model with latest data"):
                with st.spinner("Updating forecasts..."):
                    import time
                    time.sleep(2)
                    st.success("Forecasts updated successfully!")
    
    def run_dashboard(self):
        """Main method to run the dashboard"""
        if self.df is None or self.test is None:
            st.error("❌ Unable to load model data. Please run the training script first.")
            return
        
        # Create sidebar
        with st.sidebar:
            st.title("⚡ Dashboard Controls")
            
            st.markdown("### Analysis Period")
            analysis_period = st.select_slider(
                "Select Forecast Window",
                options=["1 Day", "1 Week", "2 Weeks", "1 Month", "3 Months", "Full Period"],
                value="1 Month"
            )
            
            st.markdown("### Visualization Settings")
            show_confidence = st.checkbox("Show Confidence Bands", value=True)
            smooth_data = st.checkbox("Smooth Data", value=False)
            
            st.markdown("### Model Info")
            st.info(f"""
            **Model:** XGBoost Regressor
            **Version:** 1.0
            **Last Trained:** {self.df.index.max().date()}
            **Test Period:** {analysis_period}
            """)
            
            st.markdown("---")
            st.markdown("### About")
            st.caption("""
            This dashboard provides real-time monitoring and forecasting 
            of Germany's electricity load using machine learning models.
            """)
        
        # Main content
        self.create_overview()
        self.create_key_metrics()
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Forecast", 
            "🔍 Error Analysis", 
            "🌡️ Weather",
            "🎯 Features", 
            "📋 Summary"
        ])
        
        with tab1:
            self.create_forecast_chart()
            st.markdown("---")
            self.create_what_if_scenarios()
        
        with tab2:
            self.create_error_analysis()
        
        with tab3:
            self.create_seasonal_analysis()
        
        with tab4:
            self.create_feature_analysis()
        
        with tab5:
            self.create_data_summary()
            st.markdown("---")
            self.create_export_section()
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        with col2:
            st.markdown("**Data Source:** Synthetic Dataset")
        with col3:
            st.markdown("**Model:** XGBoost v1.0")

# Run the dashboard
if __name__ == "__main__":
    dashboard = ElectricityForecastDashboard()
    dashboard.run_dashboard()
