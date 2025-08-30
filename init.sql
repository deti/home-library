-- Initialize home_library database
SELECT 'CREATE DATABASE home_library'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'home_library')\gexec
\c home_library;

-- Create extensions if they don't exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE home_library TO home_library_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO home_library_user;
