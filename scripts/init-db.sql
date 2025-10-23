-- ==========================================
-- Inicialización de Base de Datos - Rainbow Six ML
-- ==========================================

-- Crear extensiones necesarias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Crear esquemas para organizar datos
CREATE SCHEMA IF NOT EXISTS ml_models;
CREATE SCHEMA IF NOT EXISTS game_data;
CREATE SCHEMA IF NOT EXISTS experiments;

-- Tabla para almacenar datos de jugadores de Rainbow Six
CREATE TABLE IF NOT EXISTS game_data.player_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    player_name VARCHAR(100) NOT NULL,
    kdr DECIMAL(5,2),
    wlr DECIMAL(5,2),
    headshot_percentage DECIMAL(5,2),
    playtime_hours INTEGER,
    rank_level VARCHAR(20),
    operator_preference VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para almacenar métricas de modelos ML
CREATE TABLE IF NOT EXISTS ml_models.model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'clasificacion' o 'regresion'
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    mse DECIMAL(10,6),
    rmse DECIMAL(10,6),
    mae DECIMAL(10,6),
    r2_score DECIMAL(5,4),
    training_time_seconds INTEGER,
    hyperparameters JSONB,
    feature_importance JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para tracking de experimentos
CREATE TABLE IF NOT EXISTS experiments.experiment_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_name VARCHAR(100) NOT NULL,
    run_id VARCHAR(100) NOT NULL,
    pipeline_name VARCHAR(100),
    status VARCHAR(20) DEFAULT 'running', -- 'running', 'completed', 'failed'
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    parameters JSONB,
    metrics JSONB,
    artifacts JSONB,
    error_message TEXT
);

-- Índices para mejor performance
CREATE INDEX IF NOT EXISTS idx_player_stats_kdr ON game_data.player_stats(kdr);
CREATE INDEX IF NOT EXISTS idx_player_stats_rank ON game_data.player_stats(rank_level);
CREATE INDEX IF NOT EXISTS idx_model_metrics_type ON ml_models.model_metrics(model_type);
CREATE INDEX IF NOT EXISTS idx_model_metrics_name ON ml_models.model_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments.experiment_runs(experiment_name);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments.experiment_runs(status);

-- Función para actualizar timestamp automáticamente
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger para actualizar automáticamente updated_at
CREATE TRIGGER update_player_stats_updated_at 
    BEFORE UPDATE ON game_data.player_stats 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Crear usuario para la aplicación con permisos específicos
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'rainbow_six_app') THEN
        CREATE ROLE rainbow_six_app WITH LOGIN PASSWORD 'app_password_123';
    END IF;
END
$$;

-- Otorgar permisos al usuario de aplicación
GRANT USAGE ON SCHEMA ml_models TO rainbow_six_app;
GRANT USAGE ON SCHEMA game_data TO rainbow_six_app;
GRANT USAGE ON SCHEMA experiments TO rainbow_six_app;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_models TO rainbow_six_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA game_data TO rainbow_six_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA experiments TO rainbow_six_app;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_models TO rainbow_six_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA game_data TO rainbow_six_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA experiments TO rainbow_six_app;

-- Insertar algunos datos de ejemplo
INSERT INTO game_data.player_stats (player_name, kdr, wlr, headshot_percentage, playtime_hours, rank_level, operator_preference) VALUES
('Player_Alpha', 1.25, 0.68, 45.2, 150, 'Gold I', 'Ash'),
('Player_Beta', 0.89, 0.72, 38.7, 89, 'Silver III', 'Thermite'),
('Player_Gamma', 1.56, 0.81, 52.1, 245, 'Platinum II', 'Jager'),
('Player_Delta', 0.95, 0.59, 41.3, 67, 'Bronze I', 'Sledge'),
('Player_Echo', 2.13, 0.87, 61.8, 512, 'Diamond', 'Vigil')
ON CONFLICT DO NOTHING;

-- Mensaje de confirmación
DO $$
BEGIN
    RAISE NOTICE 'Base de datos inicializada correctamente para Rainbow Six ML';
    RAISE NOTICE 'Esquemas creados: ml_models, game_data, experiments';
    RAISE NOTICE 'Usuario de aplicación: rainbow_six_app';
END
$$;