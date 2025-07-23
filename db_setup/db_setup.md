## Setting up Example Database

Go to Shell / Terminal to install PostgreSQL:

**macOS (Homebrew)**

```bash
brew update
brew install postgresql
brew services start postgresql
```

Verify it is running

```python
brew services list | grep postgresql
```

Run

```python
psql -d postgres
```

You’re now in the Postgres superuser shell. Let’s create the user and database.

At the `postgres=#` prompt, type:

```sql
CREATE USER retail_admin WITH PASSWORD '<your_password>';
```

Create the `retail_store` database owned by that user

```sql
CREATE DATABASE retail_store OWNER retail_admin;
```

Then type:

```sql
GRANT ALL PRIVILEGES ON DATABASE retail_store TO retail_admin;
```

Exit psql using:

```sql
\q
```

Back in your shell prompt, run:

```sql
psql -h localhost -U retail_admin -d retail_store
```

You’ll now see:

```sql
retail_store=#
```

…and you’re ready to create tables and load data. 

At the `retail_store=#` prompt, paste and run the following SQL block:

```sql
-- 1. Create the products table
CREATE TABLE products (
  product_id        SERIAL        PRIMARY KEY,
  sku               VARCHAR(50)   UNIQUE NOT NULL,
  name              TEXT          NOT NULL,
  category          TEXT,
  aisle             VARCHAR(10)   NOT NULL,
  shelf             VARCHAR(10)   NOT NULL,
  stock_quantity    INTEGER       NOT NULL DEFAULT 0,
  reorder_threshold INTEGER       NOT NULL DEFAULT 5,
  price             NUMERIC(10,2) NOT NULL,
  supplier          TEXT,
  last_restocked    TIMESTAMP WITH TIME ZONE,
  metadata          JSONB,
  created_at        TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at        TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- 2. Speed up location lookups
CREATE INDEX idx_location ON products(aisle, shelf);

-- 3. Insert sample products
INSERT INTO products (sku, name, category, aisle, shelf, stock_quantity, price, supplier) VALUES
  ('IPH14',    'Apple iPhone 14',                  'Smartphones',    'A1','Row1',  25,  799.00, 'Apple Inc.'),
  ('SGS23U',   'Samsung Galaxy S23 Ultra',         'Smartphones',    'A1','Row2',  15, 1199.00, 'Samsung'),
  ('WH1000XM5','Sony WH-1000XM5 Headphones',       'Audio',          'B2','Row1',  30,  399.00, 'Sony'),
  ('AVOORG',   'Organic Avocados (each)',          'Produce',        'C3','Row1', 100,    2.49, 'Local Farms'),
  ('DYV15',    'Dyson V15 Detect Vacuum',           'Home Appliances','D4','Row2',  10,  749.00, 'Dyson'),
  ('LAVZSC',   'Lavazza Super Crema Coffee 2.2 lb', 'Grocery',        'E5','Row1',  40,   23.99, 'Lavazza'),
  ('IPD7',     'Instant Pot Duo 7‑in‑1',            'Kitchen',        'D1','Row3',  20,   99.95, 'Instant Brands'),
  ('MXM3S',    'Logitech MX Master 3S Mouse',       'Accessories',    'B1','Row2',  50,   99.99, 'Logitech'),
  ('HUE2P',    'Philips Hue Smart Bulbs (2‑pack)',  'Home Automation','F2','Row1',  60,   49.99, 'Philips'),
  ('TPX1C',    'Lenovo ThinkPad X1 Carbon Gen 11',  'Laptops',        'A2','Row1',   8, 1649.00, 'Lenovo'),
  ('BMBBD',    'Bamboo Cutting Board Set (3‑piece)','Kitchen',        'D2','Row1',  70,   29.95, 'EcoGoods'),
  ('TIDP3',    'Tide Pods 3‑in‑1 (81 pods)',         'Laundry',        'G1','Row1', 120,   21.99, 'Procter & Gamble');

```

This outputs:

```sql
CREATE TABLE
CREATE INDEX
INSERT 0 12
```

**Verify the data**

After running that, confirm everything loaded correctly with:

```sql
SELECT product_id, sku, name, aisle, shelf, stock_quantity, price
  FROM products
  ORDER BY product_id;
```

This outputs:

```sql
 product_id |    sku    |                name                | aisle | shelf | stock_quantity |  price
------------+-----------+------------------------------------+-------+-------+----------------+---------
          1 | IPH14     | Apple iPhone 14                    | A1    | Row1  |             25 |  799.00
          2 | SGS23U    | Samsung Galaxy S23 Ultra           | A1    | Row2  |             15 | 1199.00
          3 | WH1000XM5 | Sony WH-1000XM5 Headphones         | B2    | Row1  |             30 |  399.00
          4 | AVOORG    | Organic Avocados (each)            | C3    | Row1  |            100 |    2.49
          5 | DYV15     | Dyson V15 Detect Vacuum            | D4    | Row2  |             10 |  749.00
          6 | LAVZSC    | Lavazza Super Crema Coffee 2.2 lb  | E5    | Row1  |             40 |   23.99
          7 | IPD7      | Instant Pot Duo 7‑in‑1             | D1    | Row3  |             20 |   99.95
          8 | MXM3S     | Logitech MX Master 3S Mouse        | B1    | Row2  |             50 |   99.99
          9 | HUE2P     | Philips Hue Smart Bulbs (2‑pack)   | F2    | Row1  |             60 |   49.99
         10 | TPX1C     | Lenovo ThinkPad X1 Carbon Gen 11   | A2    | Row1  |              8 | 1649.00
         11 | BMBBD     | Bamboo Cutting Board Set (3‑piece) | D2    | Row1  |             70 |   29.95
         12 | TIDP3     | Tide Pods 3‑in‑1 (81 pods)         | G1    | Row1  |            120 |   21.99
(12 rows)
```

You should see all 12 rows listed. Once that looks good, you’re ready to connect via Python or LangChain for ingestion. 
