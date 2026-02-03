"""
Analysis Tools (UPDATED FOR LANGCHAIN)
Wraps data science functions as LangChain Tools for the Agent.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os
from langchain_core.tools import tool

# Global instance to hold state (simulating a singleton for the session)
# In a real app, this would be passed via context, but for this refactor we keep it simple.
_TOOLS_INSTANCE = None

def get_tools_instance():
    global _TOOLS_INSTANCE
    if _TOOLS_INSTANCE is None:
        _TOOLS_INSTANCE = DataAnalystTools()
    return _TOOLS_INSTANCE

class DataAnalystTools:
    def __init__(self, data_dir=None, plot_dir=None):
        # Calculate base directory relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.data_dir = data_dir if data_dir else os.path.join(base_dir, "data") + os.sep
        self.plot_dir = plot_dir if plot_dir else os.path.join(base_dir, "plots") + os.sep
            
        self.df = None
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        
        # Auto-load on init per original design
        self.load_data()

    def load_data(self):
        """Loads and merges all datasets into a Customer 360 view."""
        try:
            customers = pd.read_csv(f"{self.data_dir}customers.csv")
            orders = pd.read_csv(f"{self.data_dir}orders.csv")
            engagement = pd.read_csv(f"{self.data_dir}customer_engagement.csv")
            labels = pd.read_csv(f"{self.data_dir}churn_labels.csv")
            
            order_agg = orders.groupby('customer_id').agg({
                'order_id': 'count',
                'order_value': 'sum',
                'delivery_days': 'mean',
                'discount_applied': lambda x: (x == 'Yes').sum()
            }).reset_index()
            order_agg.columns = ['customer_id', 'total_orders', 'total_revenue', 'avg_delivery_days', 'discount_count']
            
            self.df = customers.merge(labels, on='customer_id')\
                               .merge(engagement, on='customer_id')\
                               .merge(order_agg, on='customer_id', how='left')\
                               .fillna(0)
            return "Data loaded successfully."
        except Exception as e:
            return f"Error loading data: {str(e)}"

    def get_summary_stats(self):
        if self.df is None: return "Error: Data not loaded."
        churn_rate = self.df['is_churned'].mean() * 100
        count = len(self.df)
        revenue = self.df['total_revenue'].sum()
        return f"Summary: {count} Customers, {churn_rate:.2f}% Churn Rate, Total Revenue: INR {revenue:,.0f}"

    def analyze_delivery_impact(self):
        if self.df is None: return "Error: Data not loaded."
        churned = self.df[self.df['is_churned']==1]['avg_delivery_days'].mean()
        retained = self.df[self.df['is_churned']==0]['avg_delivery_days'].mean()
        
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='is_churned', y='avg_delivery_days', data=self.df, palette=['green', 'red'])
        plt.title('Delivery Days: Retained (0) vs Churned (1)')
        path = f"{self.plot_dir}delivery_impact.png"
        plt.savefig(path)
        plt.close()
        
        return f"Delivery Analysis: Churned ({churned:.1f} days) vs Retained ({retained:.1f} days).\n[VISUALIZATION]: {path}"

    def analyze_channels(self):
        if self.df is None: return "Error: Data not loaded."
        stats = self.df.groupby('signup_channel')['is_churned'].mean().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        stats.plot(kind='bar', color='coral')
        plt.title('Churn Rate by Channel')
        path = f"{self.plot_dir}channel_churn.png"
        plt.savefig(path)
        plt.close()

        return f"Channel Analysis:\n{stats.to_string()}\n[VISUALIZATION]: {path}"

    def analyze_city_impact(self):
        if self.df is None: return "Error: Data not loaded."
        stats = self.df.groupby('city')['is_churned'].mean().sort_values(ascending=False)
        return f"Churn by City:\n{stats.to_string()}"

    def analyze_demographics(self):
        if self.df is None: return "Error: Data not loaded."
        stats = self.df.groupby('gender')['is_churned'].mean()
        return f"Churn by Gender:\n{stats.to_string()}"
    
    def analyze_engagement(self):
        if self.df is None: return "Error: Data not loaded."
        churned = self.df[self.df['is_churned']==1]['monthly_visits'].mean()
        retained = self.df[self.df['is_churned']==0]['monthly_visits'].mean()
        
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='is_churned', y='monthly_visits', data=self.df, palette=['green', 'red'])
        plt.title('Monthly Visits: Retained (0) vs Churned (1)')
        path = f"{self.plot_dir}engagement_impact.png"
        plt.savefig(path)
        plt.close()
        
        return f"Engagement: Churned ({churned:.1f}) vs Retained ({retained:.1f}) visits.\n[VISUALIZATION]: {path}"

    def train_model(self):
        if self.df is None: return "Error: Data not loaded."
        X = self.df.select_dtypes(include=['number']).drop('is_churned', axis=1)
        y = self.df['is_churned']
        clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
        
        importances = pd.Series(clf.feature_importances_, index=X.columns).nlargest(5)
        
        plt.figure(figsize=(10, 6))
        importances.plot(kind='barh', color='skyblue')
        plt.title('Top Churn Drivers')
        path = f"{self.plot_dir}feature_importance.png"
        plt.savefig(path)
        plt.close()
        
        return f"Model Trained. Top Factors:\n{importances.to_string()}\n[VISUALIZATION]: {path}"

# --- LangChain Tool Definitions ---
# These wrappers expose the class methods as tools.

@tool
def get_data_summary():
    """Get high-level summary statistics of the data (churn rate, total revenue)."""
    return get_tools_instance().get_summary_stats()

@tool
def analyze_delivery_impact():
    """Analyze and visualize if delivery time impacts churn."""
    return get_tools_instance().analyze_delivery_impact()

@tool
def analyze_channel_performance():
    """Analyze and visualize churn rates by acquisition channel."""
    return get_tools_instance().analyze_channels()

@tool
def analyze_city_performance():
    """Analyze churn rates by city/region."""
    return get_tools_instance().analyze_city_impact()

@tool
def analyze_demographics():
    """Analyze impact of gender on churn."""
    return get_tools_instance().analyze_demographics()

@tool
def analyze_engagement():
    """Analyze site visit behavior impacting churn."""
    return get_tools_instance().analyze_engagement()

@tool
def train_predictive_model():
    """Train a Machine Learning model to find the top statistical drivers of churn."""
    return get_tools_instance().train_model()
