-- Initialize e-commerce database for Text-to-SQL system
-- This script creates tables with sample data for testing

-- Create tables
CREATE TABLE IF NOT EXISTS customers (
    customer_id SERIAL PRIMARY KEY,
    customer_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    segment VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    cost DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS regions (
    region_id SERIAL PRIMARY KEY,
    region_name VARCHAR(100) NOT NULL,
    country VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    region_id INTEGER REFERENCES regions(region_id),
    order_date DATE NOT NULL,
    ship_date DATE,
    status VARCHAR(50) NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    total_price DECIMAL(10, 2) NOT NULL
);

CREATE TABLE IF NOT EXISTS sales (
    sale_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    customer_id INTEGER REFERENCES customers(customer_id),
    sale_date DATE NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    quantity INTEGER NOT NULL,
    profit DECIMAL(10, 2)
);

-- Create indexes for performance
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_sales_date ON sales(sale_date);
CREATE INDEX idx_sales_product ON sales(product_id);
CREATE INDEX idx_sales_customer ON sales(customer_id);
CREATE INDEX idx_products_category ON products(category);

-- Create read-only user
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'readonly_user') THEN
        CREATE ROLE readonly_user WITH LOGIN PASSWORD 'readonly_password';
    END IF;
END
$$;

-- Grant read-only permissions
GRANT CONNECT ON DATABASE ecommerce TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly_user;

-- Insert sample data (subset for demo - full dataset would be 1.55M+ records)
INSERT INTO customers (customer_name, email, segment) VALUES
('John Smith', 'john@example.com', 'Consumer'),
('Jane Doe', 'jane@example.com', 'Corporate'),
('Bob Johnson', 'bob@example.com', 'Consumer'),
('Alice Williams', 'alice@example.com', 'Corporate'),
('Charlie Brown', 'charlie@example.com', 'Home Office');

INSERT INTO products (product_name, category, price, cost) VALUES
('Laptop', 'Electronics', 999.99, 600.00),
('Mouse', 'Electronics', 29.99, 10.00),
('Keyboard', 'Electronics', 79.99, 30.00),
('Monitor', 'Electronics', 399.99, 250.00),
('Desk Chair', 'Furniture', 199.99, 120.00),
('Standing Desk', 'Furniture', 499.99, 300.00),
('Notebook', 'Office Supplies', 5.99, 2.00),
('Pen Set', 'Office Supplies', 12.99, 5.00);

INSERT INTO regions (region_name, country) VALUES
('West', 'United States'),
('East', 'United States'),
('Central', 'United States'),
('South', 'United States'),
('Europe', 'Germany'),
('Asia', 'Japan');

-- Insert orders with various dates
INSERT INTO orders (customer_id, region_id, order_date, ship_date, status, total_amount) VALUES
(1, 1, '2024-01-15', '2024-01-17', 'Shipped', 999.99),
(2, 2, '2024-01-16', '2024-01-18', 'Shipped', 129.98),
(3, 3, '2024-01-17', '2024-01-19', 'Delivered', 199.99),
(4, 4, '2024-01-18', '2024-01-20', 'Shipped', 399.99),
(5, 1, '2024-01-19', '2024-01-21', 'Processing', 499.99);

-- Insert order items
INSERT INTO order_items (order_id, product_id, quantity, unit_price, total_price) VALUES
(1, 1, 1, 999.99, 999.99),
(2, 2, 1, 29.99, 29.99),
(2, 3, 1, 79.99, 79.99),
(3, 5, 1, 199.99, 199.99),
(4, 4, 1, 399.99, 399.99),
(5, 6, 1, 499.99, 499.99);

-- Insert sales data
INSERT INTO sales (order_id, product_id, customer_id, sale_date, amount, quantity, profit) VALUES
(1, 1, 1, '2024-01-15', 999.99, 1, 399.99),
(2, 2, 2, '2024-01-16', 29.99, 1, 19.99),
(2, 3, 2, '2024-01-16', 79.99, 1, 49.99),
(3, 5, 3, '2024-01-17', 199.99, 1, 79.99),
(4, 4, 4, '2024-01-18', 399.99, 1, 149.99),
(5, 6, 5, '2024-01-19', 499.99, 1, 199.99);

-- Generate additional sales data for testing (would be 1.55M+ in production)
INSERT INTO sales (order_id, product_id, customer_id, sale_date, amount, quantity, profit)
SELECT
    (random() * 5 + 1)::int as order_id,
    (random() * 7 + 1)::int as product_id,
    (random() * 5 + 1)::int as customer_id,
    CURRENT_DATE - (random() * 365)::int as sale_date,
    (random() * 1000 + 10)::decimal(10,2) as amount,
    (random() * 10 + 1)::int as quantity,
    (random() * 500)::decimal(10,2) as profit
FROM generate_series(1, 1000);

-- Update statistics
ANALYZE customers;
ANALYZE products;
ANALYZE regions;
ANALYZE orders;
ANALYZE order_items;
ANALYZE sales;
