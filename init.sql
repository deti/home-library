-- Initialize home_library database
CREATE DATABASE IF NOT EXISTS home_library;
\c home_library;

-- Create extensions if they don't exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE home_library TO home_library_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO home_library_user;
