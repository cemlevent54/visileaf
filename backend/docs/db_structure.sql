-- Veritabanı Yapısı
-- PostgreSQL için optimize edilmiş
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    deleted_at TIMESTAMP DEFAULT NULL
);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_deleted_at ON users(deleted_at)
WHERE deleted_at IS NULL;
CREATE TABLE password_reset_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_password_reset_tokens_user_id ON password_reset_tokens(user_id);
CREATE INDEX idx_password_reset_tokens_token_hash ON password_reset_tokens(token_hash);
CREATE INDEX idx_password_reset_tokens_expires_at ON password_reset_tokens(expires_at)
WHERE used = FALSE;
CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    parent_image_id UUID REFERENCES images(id) ON DELETE CASCADE,
    -- NULL = orijinal görsel
    -- dolu = işlenmiş sonuç
    file_path TEXT NOT NULL,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    enhancement_type VARCHAR(50),
    -- 'clahe', 'gamma', 'ssr', 'msr', 'sharpen', 'hybrid' vb.
    params JSONB,
    -- Enhancement parametreleri
    -- Örnek params yapısı:
    -- {
    --   "methods": ["clahe", "gamma"],
    --   "order": ["gamma", "clahe"],
    --   "clahe": {"clip_limit": 3.0, "tile_size": [8, 8]},
    --   "gamma": {"value": 0.5},
    --   "ssr": {"sigma": 80},
    --   "msr": {"sigmas": [15, 80, 250]},
    --   "sharpen": {"method": "unsharp", "strength": 1.0, "kernel_size": 5}
    -- }
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_images_user_id ON images(user_id);
CREATE INDEX idx_images_parent_image_id ON images(parent_image_id);
CREATE INDEX idx_images_enhancement_type ON images(enhancement_type);
CREATE INDEX idx_images_created_at ON images(created_at);
-- JSONB index for params queries
CREATE INDEX idx_images_params ON images USING GIN (params);
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE
    SET NULL,
        action VARCHAR(255) NOT NULL,
        details JSONB,
        -- metadata yerine details kullanıyoruz (SQLAlchemy'de metadata rezervli)
        created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX idx_audit_logs_details ON audit_logs USING GIN (details);