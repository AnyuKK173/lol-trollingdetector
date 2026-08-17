CREATE TABLE IF NOT EXISTS rank_snapshots (
    id BIGSERIAL PRIMARY KEY,
    puuid TEXT NOT NULL,
    queue_type TEXT NOT NULL,
    tier TEXT NOT NULL,
    division TEXT NOT NULL,
    league_points INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rank_snapshots_player_time
    ON rank_snapshots (puuid, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_rank_snapshots_tier_division_time
    ON rank_snapshots (tier, division, observed_at DESC);

CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    data_version TEXT,
    platform_id TEXT,
    regional_route TEXT NOT NULL,
    queue_id INTEGER NOT NULL,
    game_version TEXT,
    patch TEXT,
    map_id INTEGER,
    game_mode TEXT,
    game_type TEXT,
    game_start TIMESTAMPTZ,
    game_end TIMESTAMPTZ,
    duration_seconds INTEGER,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    collection_status TEXT NOT NULL DEFAULT 'collecting',
    collection_error TEXT,
    CONSTRAINT matches_collection_status_check
        CHECK (collection_status IN ('collecting', 'complete', 'failed'))
);

-- Migration: allow a 'skipped' status for matches that were deliberately
-- rejected (patch mismatch, rank drift) rather than failed due to an error.
-- CREATE TABLE IF NOT EXISTS above does not alter an already-existing table,
-- so the constraint has to be widened explicitly and idempotently.
DO $$
BEGIN
    ALTER TABLE matches DROP CONSTRAINT IF EXISTS matches_collection_status_check;
    ALTER TABLE matches ADD CONSTRAINT matches_collection_status_check
        CHECK (collection_status IN ('collecting', 'complete', 'failed', 'skipped'));
END $$;

CREATE INDEX IF NOT EXISTS idx_matches_patch_queue
    ON matches (patch, queue_id);
CREATE INDEX IF NOT EXISTS idx_matches_status
    ON matches (collection_status);

CREATE TABLE IF NOT EXISTS participants (
    match_id TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    participant_id INTEGER NOT NULL,
    puuid TEXT,
    team_id INTEGER,
    team_position TEXT,
    individual_position TEXT,
    champion_id INTEGER,
    champion_name TEXT,
    win BOOLEAN,
    kills INTEGER,
    deaths INTEGER,
    assists INTEGER,
    gold_earned INTEGER,
    total_minions_killed INTEGER,
    neutral_minions_killed INTEGER,
    vision_score INTEGER,
    wards_placed INTEGER,
    wards_killed INTEGER,
    damage_to_champions INTEGER,
    damage_taken INTEGER,
    damage_to_objectives INTEGER,
    damage_to_turrets INTEGER,
    time_ccing_others INTEGER,
    total_time_cc_dealt INTEGER,
    PRIMARY KEY (match_id, participant_id)
);

CREATE INDEX IF NOT EXISTS idx_participants_role_champion
    ON participants (team_position, champion_id);
CREATE INDEX IF NOT EXISTS idx_participants_puuid
    ON participants (puuid);

CREATE TABLE IF NOT EXISTS participant_frames (
    match_id TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    participant_id INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    current_gold INTEGER,
    total_gold INTEGER,
    level INTEGER,
    xp INTEGER,
    minions_killed INTEGER,
    jungle_minions_killed INTEGER,
    position_x INTEGER,
    position_y INTEGER,
    time_enemy_spent_controlled INTEGER,
    champion_stats JSONB NOT NULL DEFAULT '{}'::jsonb,
    damage_stats JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (match_id, participant_id, timestamp_ms),
    FOREIGN KEY (match_id, participant_id)
        REFERENCES participants(match_id, participant_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_frames_minute_role_join
    ON participant_frames (minute, match_id, participant_id);

CREATE TABLE IF NOT EXISTS timeline_events (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    timestamp_ms INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    participant_id INTEGER,
    killer_id INTEGER,
    victim_id INTEGER,
    creator_id INTEGER,
    assisting_participant_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    team_id INTEGER,
    item_id INTEGER,
    ward_type TEXT,
    monster_type TEXT,
    monster_sub_type TEXT,
    building_type TEXT,
    lane_type TEXT,
    tower_type TEXT,
    position_x INTEGER,
    position_y INTEGER,
    raw_event JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_match_time
    ON timeline_events (match_id, timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_events_type_time
    ON timeline_events (event_type, timestamp_ms);

-- Persists discovered Gold ladder players across runs so re-running the
-- collector continues the same subject list instead of re-drawing a fresh
-- random sample and losing track of who has already been worked through.
CREATE TABLE IF NOT EXISTS collection_subjects (
    puuid TEXT PRIMARY KEY,
    tier TEXT NOT NULL,
    division TEXT NOT NULL,
    discovered_at TIMESTAMPTZ NOT NULL,
    last_attempted_at TIMESTAMPTZ,
    accepted_matches INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    CONSTRAINT collection_subjects_status_check
        CHECK (status IN ('pending', 'in_progress', 'exhausted', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_collection_subjects_status
    ON collection_subjects (status, last_attempted_at);

-- One row per collector.py invocation, for observability of what each run
-- was targeting and how it landed (accepted vs. patch-mismatched vs. failed).
CREATE TABLE IF NOT EXISTS collection_runs (
    run_id BIGSERIAL PRIMARY KEY,
    target_patch TEXT NOT NULL,
    target_queue INTEGER NOT NULL,
    target_verified_matches INTEGER NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    accepted_matches INTEGER NOT NULL DEFAULT 0,
    patch_mismatch_matches INTEGER NOT NULL DEFAULT 0,
    failed_matches INTEGER NOT NULL DEFAULT 0
);
