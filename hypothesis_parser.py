"""
Natural Language Hypothesis Parser
Converts plain English trading hypotheses into executable code
"""

import re
import numpy as np
import pandas as pd


class HypothesisParser:
    def __init__(self, df):
        """Initialize with dataframe containing market data"""
        self.df = df.copy()
        self.variables = {}
        
    def parse(self, hypothesis_text, timeframe='daily'):
        """
        Parse plain English hypothesis into executable code
        
        Examples:
        - "nifty return is greater than 1%"
        - "vix is above 20"
        - "last month return is positive and this month return is also positive"
        - "nifty return is more than mean + 2 standard deviations"
        """
        
        # Normalize text
        text = hypothesis_text.lower().strip()
        
        # Calculate common statistics
        self._calculate_statistics(timeframe)
        
        # Try different parsing strategies
        result = None
        
        # Strategy 1: Multi-period comparisons (last/previous month vs this/current month)
        period_keywords = ['month', 'week', 'day']
        temporal_keywords = ['last', 'previous', 'prior', 'past']
        
        if any(pk in text for pk in period_keywords) and any(tk in text for tk in temporal_keywords):
            result = self._parse_multi_period(text, timeframe)
        
        # Strategy 2: Statistical comparisons (mean, std dev, etc.)
        elif any(word in text for word in ['mean', 'average', 'median', 'std', 'standard deviation']):
            result = self._parse_statistical(text, timeframe)
        
        # Strategy 3: Simple threshold comparisons
        else:
            result = self._parse_simple(text, timeframe)
        
        if result is None:
            raise ValueError(f"Could not parse hypothesis: {hypothesis_text}")
        
        return result, self.variables
    
    def _calculate_statistics(self, timeframe):
        """Pre-calculate common statistics"""
        df = self.df
        
        # Ensure we have Date column
        if 'Date' not in df.columns:
            if df.index.name == 'Date':
                df = df.reset_index()
            else:
                raise ValueError("DataFrame must have a 'Date' column")
        
        # Ensure Close_nifty and Close_vix exist
        if 'Close_nifty' not in df.columns:
            raise ValueError("'Close_nifty' column not found in dataframe")
        if 'Close_vix' not in df.columns:
            raise ValueError("'Close_vix' column not found in dataframe")
        
        # Calculate returns if not present
        if 'nifty_return' not in df.columns:
            df['nifty_return'] = df['Close_nifty'].pct_change() * 100
        if 'vix_change' not in df.columns:
            df['vix_change'] = df['Close_vix'].diff()
        if 'vix_pct_change' not in df.columns:
            df['vix_pct_change'] = df['Close_vix'].pct_change() * 100
        
        # Statistics
        self.variables['mean_nifty_return'] = df['nifty_return'].mean()
        self.variables['std_nifty_return'] = df['nifty_return'].std()
        self.variables['median_nifty_return'] = df['nifty_return'].median()
        
        self.variables['mean_vix'] = df['Close_vix'].mean()
        self.variables['std_vix'] = df['Close_vix'].std()
        self.variables['median_vix'] = df['Close_vix'].median()
        
        self.variables['mean_vix_change'] = df['vix_change'].mean()
        self.variables['std_vix_change'] = df['vix_change'].std()
        
        self.df = df
    
    def _parse_multi_period(self, text, timeframe):
        """
        Parse multi-period comparisons
        Example: "last month return is more than mean+2sd and this month return is also mean+2sd"
        
        For month/week comparisons, we work at the period level and then expand back to daily
        """
        df = self.df.copy()
        
        # Ensure we have nifty_return
        if 'nifty_return' not in df.columns:
            df['nifty_return'] = df['Close_nifty'].pct_change() * 100
        
        # Determine period type from text
        if 'month' in text:
            period = 'month'
            df['period'] = df['Date'].dt.to_period('M')
        elif 'week' in text:
            period = 'week'
            df['period'] = df['Date'].dt.to_period('W')
        else:  # day
            period = 'day'
            # For daily, just use the date itself
            df['period_return'] = df['nifty_return']
            df['last_period_return'] = df['nifty_return'].shift(1)
            df = df.fillna(0)
            
            # Parse conditions for daily
            last_condition = None
            current_condition = None
            
            # Define temporal keywords for flexibility
            last_keywords = ['last', 'previous', 'prior', 'past']
            current_keywords = ['this', 'current', 'present']
            
            has_last_keyword = any(kw in text for kw in last_keywords)
            has_current_keyword = any(kw in text for kw in current_keywords)
            
            if 'positive' in text or 'greater than 0' in text or '> 0' in text:
                if has_last_keyword:
                    last_condition = df['last_period_return'] > 0
                if has_current_keyword or ('also' in text and 'positive' in text):
                    current_condition = df['period_return'] > 0
            elif 'negative' in text or 'less than 0' in text or '< 0' in text:
                if has_last_keyword:
                    last_condition = df['last_period_return'] < 0
                if has_current_keyword or ('also' in text and 'negative' in text):
                    current_condition = df['period_return'] < 0
            
            # Store for later use
            self.df = df
            
            if last_condition is not None and current_condition is not None:
                if 'and' in text or 'also' in text:
                    return last_condition & current_condition
                elif 'or' in text:
                    return last_condition | current_condition
                else:
                    return last_condition & current_condition
            elif last_condition is not None:
                return last_condition
            elif current_condition is not None:
                return current_condition
            return None
        
        # For week/month - work at period level, then broadcast to daily
        # Calculate proper period returns using first and last Close
        period_data = df.groupby('period').agg({
            'Close_nifty': ['first', 'last'],
            'Date': 'last'
        }).reset_index()
        
        # Flatten column names
        period_data.columns = ['period', 'Close_first', 'Close_last', 'Date']
        
        # Calculate period return as percentage change from first to last price
        period_data['period_return'] = ((period_data['Close_last'] / period_data['Close_first']) - 1) * 100
        
        # Calculate last period return
        period_data['last_period_return'] = period_data['period_return'].shift(1)
        
        # Calculate statistics on period returns (excluding first period where last_period_return is NaN)
        period_data_valid = period_data.dropna(subset=['last_period_return']).copy()
        mean_period_return = period_data_valid['period_return'].mean()
        std_period_return = period_data_valid['period_return'].std()
        
        # NOW evaluate conditions at the PERIOD level, not daily level
        last_condition_period = None
        current_condition_period = None
        
        # Define temporal keywords for flexibility
        last_keywords = ['last', 'previous', 'prior', 'past']
        current_keywords = ['this', 'current', 'present']
        
        has_last_keyword = any(kw in text for kw in last_keywords)
        has_current_keyword = any(kw in text for kw in current_keywords)
        
        # Debug: show what we're looking for
        print(f"DEBUG: Parsing text: '{text}'")
        print(f"DEBUG: has_last_keyword={has_last_keyword}, has_current_keyword={has_current_keyword}")
        print(f"DEBUG: 'positive' in text={('positive' in text)}, 'negative' in text={('negative' in text)}")
        
        # Look for patterns like "mean + 2sd" or "mean+2*std"
        if 'mean' in text and ('sd' in text or 'std' in text or 'standard deviation' in text):
            # Extract multiplier
            sd_match = re.search(r'(\d+\.?\d*)\s*(?:sd|std|standard deviation)', text)
            multiplier = float(sd_match.group(1)) if sd_match else 2.0
            
            threshold = mean_period_return + multiplier * std_period_return
            
            if has_last_keyword:
                # For periods without last data, mark as False
                last_condition_period = period_data['last_period_return'].fillna(-999999) > threshold
            
            if has_current_keyword:
                current_condition_period = period_data['period_return'] > threshold
        
        # Look for positive/negative conditions
        elif 'positive' in text or 'greater than 0' in text or '> 0' in text:
            if has_last_keyword:
                # For periods without last data (first period), mark as False
                last_condition_period = period_data['last_period_return'].fillna(-1) > 0
            if has_current_keyword or ('also' in text and 'positive' in text):
                current_condition_period = period_data['period_return'] > 0
        
        elif 'negative' in text or 'less than 0' in text or '< 0' in text:
            if has_last_keyword:
                # For periods without last data (first period), mark as False
                last_condition_period = period_data['last_period_return'].fillna(1) < 0
            if has_current_keyword or ('also' in text and 'negative' in text):
                current_condition_period = period_data['period_return'] < 0
        
        # Look for "same direction" or "same sign" patterns
        elif 'same' in text and ('direction' in text or 'sign' in text):
            # Both positive OR both negative
            both_positive = (period_data['last_period_return'].fillna(-1) > 0) & (period_data['period_return'] > 0)
            both_negative = (period_data['last_period_return'].fillna(1) < 0) & (period_data['period_return'] < 0)
            # Combine with OR - true if either both positive or both negative
            period_data['hypothesis_result'] = both_positive | both_negative
            
            # Debug output
            print(f"DEBUG: Sample period data (first 10 periods):")
            print(period_data[['period', 'period_return', 'last_period_return', 'hypothesis_result']].head(10))
            print(f"DEBUG: Hypothesis TRUE for {period_data['hypothesis_result'].sum()} periods out of {len(period_data)}")
            
            # Broadcast to daily rows
            df = df.merge(period_data[['period', 'period_return', 'last_period_return', 'hypothesis_result']], 
                          on='period', how='left')
            self.df = df
            return df['hypothesis_result'].values
        
        # Combine period-level conditions
        if last_condition_period is not None and current_condition_period is not None:
            if 'and' in text or 'also' in text:
                period_data['hypothesis_result'] = last_condition_period & current_condition_period
            elif 'or' in text:
                period_data['hypothesis_result'] = last_condition_period | current_condition_period
            else:
                period_data['hypothesis_result'] = last_condition_period & current_condition_period
        elif last_condition_period is not None:
            period_data['hypothesis_result'] = last_condition_period
        elif current_condition_period is not None:
            period_data['hypothesis_result'] = current_condition_period
        else:
            period_data['hypothesis_result'] = False
        
        # Debug output
        print(f"DEBUG: Sample period data (first 10 periods):")
        print(period_data[['period', 'period_return', 'last_period_return', 'hypothesis_result']].head(10))
        print(f"DEBUG: Hypothesis TRUE for {period_data['hypothesis_result'].sum()} periods out of {len(period_data)}")
        
        # Now broadcast the period-level result back to all daily rows in that period
        df = df.merge(period_data[['period', 'period_return', 'last_period_return', 'hypothesis_result']], 
                      on='period', how='left')
        
        # Store the modified df
        self.df = df
        
        # Return the daily-level boolean array
        return df['hypothesis_result'].values
    
    def _parse_statistical(self, text, timeframe):
        """
        Parse statistical comparisons
        Example: "nifty return is more than mean + 2 standard deviations"
        """
        df = self.df
        
        # Determine what variable we're testing
        if 'nifty' in text and 'return' in text:
            variable = 'nifty_return'
            mean = self.variables['mean_nifty_return']
            std = self.variables['std_nifty_return']
            median = self.variables['median_nifty_return']
            values = df['nifty_return'].values
        elif 'vix' in text and 'change' in text:
            variable = 'vix_change'
            mean = self.variables['mean_vix_change']
            std = self.variables['std_vix_change']
            median = df['vix_change'].median()
            values = df['vix_change'].values
        elif 'vix' in text:
            variable = 'vix'
            mean = self.variables['mean_vix']
            std = self.variables['std_vix']
            median = self.variables['median_vix']
            values = df['Close_vix'].values
        else:
            return None
        
        # Parse statistical threshold
        threshold = mean  # default
        
        # Look for mean +/- N * std
        if 'mean' in text:
            threshold = mean
            
            # Check for standard deviation adjustment
            if 'sd' in text or 'std' in text or 'standard deviation' in text:
                # Extract multiplier
                sd_match = re.search(r'(\d+\.?\d*)\s*(?:\*\s*)?(?:sd|std|standard deviation)', text)
                if not sd_match:
                    sd_match = re.search(r'(?:mean|average)\s*\+\s*(\d+\.?\d*)', text)
                
                multiplier = float(sd_match.group(1)) if sd_match else 2.0
                
                if '+' in text or 'plus' in text or 'above' in text or 'more than' in text or 'greater than' in text:
                    threshold = mean + multiplier * std
                elif '-' in text or 'minus' in text or 'below' in text or 'less than' in text:
                    threshold = mean - multiplier * std
        
        elif 'median' in text:
            threshold = median
        
        # Determine comparison operator
        if any(word in text for word in ['greater than', 'more than', 'above', 'exceeds', '>']):
            return values > threshold
        elif any(word in text for word in ['less than', 'below', 'under', '<']):
            return values < threshold
        elif any(word in text for word in ['equals', 'equal to', '==']):
            return np.abs(values - threshold) < 0.01
        else:
            return values > threshold  # default to greater than
    
    def _parse_simple(self, text, timeframe):
        """
        Parse simple threshold comparisons
        Example: "nifty return is greater than 1%", "vix is above 20"
        """
        df = self.df
        
        # Extract numeric threshold
        number_match = re.search(r'(\d+\.?\d*)\s*%?', text)
        if not number_match:
            return None
        
        threshold = float(number_match.group(1))
        
        # Determine variable
        if 'nifty' in text and 'return' in text:
            values = df['nifty_return'].values
        elif 'vix' in text and 'change' in text:
            values = df['vix_change'].values
        elif 'vix' in text:
            values = df['Close_vix'].values
        elif 'nifty' in text:
            values = df['Close_nifty'].values
        else:
            return None
        
        # Determine comparison operator
        if any(word in text for word in ['greater than', 'more than', 'above', 'exceeds', '>', 'over']):
            return values > threshold
        elif any(word in text for word in ['less than', 'below', 'under', '<']):
            return values < threshold
        elif any(word in text for word in ['equals', 'equal to', '==', 'is']):
            # For "is" check if we have a range
            if 'between' in text:
                # Extract range
                range_match = re.findall(r'(\d+\.?\d*)', text)
                if len(range_match) >= 2:
                    low = float(range_match[0])
                    high = float(range_match[1])
                    return (values >= low) & (values <= high)
            return np.abs(values - threshold) < 0.01
        elif 'positive' in text:
            return values > 0
        elif 'negative' in text:
            return values < 0
        else:
            return values > threshold  # default


def test_parser():
    """Test the parser with sample data"""
    # Create sample data
    dates = pd.date_range('2015-01-01', '2025-01-01', freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'Date': dates,
        'Close_nifty': 10000 + np.cumsum(np.random.randn(len(dates)) * 100),
        'Close_vix': 15 + np.random.randn(len(dates)) * 3
    })
    
    parser = HypothesisParser(df)
    
    test_cases = [
        "nifty return is greater than 1%",
        "vix is above 20",
        "nifty return is more than mean + 2 standard deviations",
        "vix change is positive",
    ]
    
    for test in test_cases:
        try:
            result, variables = parser.parse(test)
            print(f"✓ '{test}' -> {result.sum()} True out of {len(result)}")
        except Exception as e:
            print(f"✗ '{test}' -> Error: {e}")


if __name__ == "__main__":
    test_parser()
