import psycopg2
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

class PostgresHandler:
    def __init__(self):
        self.database=os.getenv('SQL_DATABASE')
        self.user=os.getenv('SQL_USER')
        self.host=os.getenv('SQL_HOST')
        self.password=os.getenv('SQL_PASSWORD')
        self.port=os.getenv('SQL_PORT')
        self.connection = None
        self.columns_to_encode = ['concord_id', 'data_type', 'metric', 'unit', 'device_type', 
                                  'vendor', 'model', 'firmware', 'name', 'reference']
        self.encoding_map = {}

    def connect(self):
        """Establish a connection to the PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(
                database=self.database,
                user=self.user,
                host=self.host,
                password=self.password,
                port=self.port
            )
            print("Connection to PostgreSQL established.")
        except Exception as e:
            print(f"Error connecting to database: {e}")

    def disconnect(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print("PostgreSQL connection closed.")

    def execute_query(self, query, params=None):
        """Execute a query and return the result as a DataFrame."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                return pd.DataFrame(results, columns=columns)
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    def get_data(self, table_name, columns, limit=None, encode=False):
        """Fetch data from a table, with optional encoding of specific columns."""
        columns_str = ', '.join(columns)
        
        # Construct the base query
        query = f"SELECT {columns_str} FROM {table_name}"
        if limit is not None:
            query += f" LIMIT {limit}"
        
        # Execute the query and retrieve the DataFrame
        df = self.execute_query(query)
        
        if df is not None and encode:
            # OneHotEncode the columns specified in `self.columns_to_encode`
            for column in self.columns_to_encode:
                if column in df.columns:
                    # Apply OneHotEncoder to the column
                    one_hot_encoder = OneHotEncoder(sparse_output=False)
                    encoded_cols = one_hot_encoder.fit_transform(df[[column]])
                    
                    # Store the encoding map
                    categories = one_hot_encoder.categories_[0]
                    self.encoding_map[column] = {idx: category for idx, category in enumerate(categories)}
                    
                    # Replace the original column with integer-encoded values
                    df[column] = encoded_cols.argmax(axis=1)
                    
        return df

# Example usage
if __name__ == "__main__":
    handler = PostgresHandler(
        database="nutanix",
        user="postgres",
        host='172.25.221.34',
        password="Senna",
        port=1433
    )
    
    handler.connect()
    clean_data_df = handler.get_data(
        table_name="ssd_clean_data", 
        columns=["data_type", "metric", "unit", "other_column"], 
        limit=100, 
        encode=True
    )
    if clean_data_df is not None:
        print(clean_data_df)
        print("\nEncoding Map:")
        print(handler.encoding_map)  # To view the mapping of encoded integers to original categories
    handler.disconnect()
