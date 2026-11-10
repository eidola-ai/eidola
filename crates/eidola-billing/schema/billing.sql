-- Billing operations. This block is embedded verbatim in the gateway baseline.
-- All state here belongs to the linked account surface, never to ACT spends.
ALTER TABLE account ADD COLUMN billing_state TEXT NOT NULL DEFAULT 'open'
    CHECK (billing_state IN ('open', 'closing', 'closed'));
ALTER TABLE account ADD COLUMN billing_closed_at TIMESTAMPTZ;

CREATE TABLE billing_operation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES account(id),
    kind TEXT NOT NULL CHECK (kind IN ('purchase', 'refund', 'dispute', 'closure')),
    actor TEXT NOT NULL DEFAULT current_user,
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(id,account_id)
);

CREATE TABLE billing_purchase (
    id TEXT PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES account(id),
    operation_id UUID NOT NULL UNIQUE REFERENCES billing_operation(id),
    charge_id TEXT UNIQUE,
    payment_intent TEXT UNIQUE,
    invoice_id TEXT UNIQUE,
    currency TEXT NOT NULL CHECK (currency ~ '^[a-z]{3}$'),
    paid_minor BIGINT NOT NULL CHECK (paid_minor >= 0),
    credits BIGINT NOT NULL CHECK (credits > 0),
    expires_at TIMESTAMPTZ NOT NULL,
    purchased_at TIMESTAMPTZ NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('purchase', 'subscription_renewal')),
    CHECK (paid_minor = 0 OR charge_id IS NOT NULL),
    CHECK (kind = 'subscription_renewal' OR expires_at > purchased_at),
    UNIQUE(id,account_id),
    UNIQUE(id,expires_at)
);

ALTER TABLE credit_ledger ADD COLUMN purchase_id TEXT REFERENCES billing_purchase(id);
ALTER TABLE credit_ledger ADD COLUMN billing_operation_id UUID REFERENCES billing_operation(id);
ALTER TABLE credit_ledger ADD CONSTRAINT payment_requires_operation CHECK (
    reason NOT IN ('purchase', 'subscription_renewal', 'refund', 'dispute_clawback', 'dispute_reversal')
    OR (billing_operation_id IS NOT NULL AND purchase_id IS NOT NULL)
);
ALTER TABLE credit_ledger ADD FOREIGN KEY(purchase_id,account_id) REFERENCES billing_purchase(id,account_id);
ALTER TABLE credit_ledger ADD FOREIGN KEY(billing_operation_id,account_id) REFERENCES billing_operation(id,account_id);
ALTER TABLE credit_ledger ADD FOREIGN KEY(purchase_id,expires_at) REFERENCES billing_purchase(id,expires_at);
ALTER TABLE credit_ledger ADD CONSTRAINT purchase_expiry_required CHECK (purchase_id IS NULL OR expires_at IS NOT NULL);
CREATE INDEX idx_ledger_purchase ON credit_ledger(purchase_id);
CREATE INDEX idx_ledger_operation ON credit_ledger(billing_operation_id);

CREATE TABLE billing_refund_operation (
    id UUID PRIMARY KEY REFERENCES billing_operation(id),
    purchase_id TEXT NOT NULL REFERENCES billing_purchase(id),
    credits BIGINT NOT NULL CHECK (credits >= 0),
    amount_minor BIGINT NOT NULL CHECK (amount_minor >= 0),
    state TEXT NOT NULL DEFAULT 'prepared' CHECK (state IN (
        'prepared', 'submitting', 'pending', 'requires_action', 'succeeded',
        'failed', 'canceled', 'released', 'superseded')),
    refund_id TEXT UNIQUE,
    submitted_at TIMESTAMPTZ,
    replaces UUID UNIQUE REFERENCES billing_refund_operation(id)
);

CREATE TABLE billing_external_refund (
    operation_id UUID PRIMARY KEY REFERENCES billing_refund_operation(id),
    reference TEXT NOT NULL UNIQUE,
    actor TEXT NOT NULL DEFAULT current_user,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE billing_refund_observation (
    id TEXT PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES account(id),
    charge_id TEXT NOT NULL,
    amount_minor BIGINT NOT NULL CHECK (amount_minor > 0),
    currency TEXT NOT NULL CHECK (currency ~ '^[a-z]{3}$'),
    status TEXT NOT NULL CHECK (status IN ('pending', 'requires_action', 'succeeded', 'failed', 'canceled')),
    operation_id UUID UNIQUE REFERENCES billing_refund_operation(id),
    observed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

-- Financial facts whose purchase or operation provenance needs human review.
-- These have no credit effect. Keeping them must not block unrelated purchases.
CREATE TABLE billing_unmatched_observation (
    account_id UUID NOT NULL REFERENCES account(id),
    kind TEXT NOT NULL CHECK (kind IN ('refund', 'dispute')),
    id TEXT NOT NULL,
    charge_id TEXT NOT NULL,
    amount_minor BIGINT NOT NULL CHECK (amount_minor > 0),
    currency TEXT NOT NULL,
    status TEXT NOT NULL,
    operation_id UUID,
    reason TEXT NOT NULL,
    PRIMARY KEY(kind,id)
);

CREATE TABLE billing_dispute (
    id TEXT PRIMARY KEY,
    purchase_id TEXT NOT NULL REFERENCES billing_purchase(id),
    operation_id UUID NOT NULL UNIQUE REFERENCES billing_operation(id),
    amount_minor BIGINT NOT NULL CHECK (amount_minor > 0),
    status TEXT NOT NULL CHECK (status IN ('needs_response', 'under_review', 'lost', 'won',
        'warning_needs_response', 'warning_under_review', 'warning_closed')),
    target_minor NUMERIC NOT NULL DEFAULT 0 CHECK (target_minor >= 0),
    debt_minor NUMERIC NOT NULL DEFAULT 0 CHECK (debt_minor >= 0)
);

-- The exact allocations a reversal must undo, including their original expiry.
CREATE TABLE billing_dispute_effect (
    ledger_id UUID PRIMARY KEY REFERENCES credit_ledger(id),
    dispute_id TEXT NOT NULL REFERENCES billing_dispute(id),
    purchase_id TEXT NOT NULL REFERENCES billing_purchase(id),
    credits BIGINT NOT NULL CHECK (credits > 0),
    restored_credits BIGINT NOT NULL DEFAULT 0 CHECK (restored_credits >= 0 AND restored_credits <= credits),
    cash_minor NUMERIC NOT NULL CHECK (cash_minor > 0)
);

CREATE TABLE billing_event (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    completed_at TIMESTAMPTZ,
    last_error TEXT,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    received_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX idx_billing_event_pending ON billing_event(received_at) WHERE completed_at IS NULL;

-- Purchase-level allocation replaces pools which accidentally merged prices.
-- This signature is also called by record_credential_issuance.
CREATE OR REPLACE FUNCTION debit_account(
    p_account_id UUID, p_amount BIGINT, p_reason TEXT,
    p_stripe_event_id TEXT DEFAULT NULL, p_credential_key_id TEXT DEFAULT NULL,
    p_require_balance BOOLEAN DEFAULT TRUE, p_payment_intent TEXT DEFAULT NULL,
    p_memo TEXT DEFAULT NULL
) RETURNS UUID[] LANGUAGE plpgsql AS $$
DECLARE
    v_remaining BIGINT := p_amount;
    v_pool RECORD;
    v_take BIGINT;
    v_id UUID;
    v_ids UUID[] := '{}';
    v_state TEXT;
BEGIN
    SELECT billing_state INTO STRICT v_state FROM account WHERE id = p_account_id FOR UPDATE;
    IF p_amount <= 0 OR p_reason NOT IN ('credential_issuance', 'manual_adjustment') THEN
        RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'use a billing operation for payment adjustments';
    END IF;
    IF p_reason = 'credential_issuance' AND v_state <> 'open' THEN RETURN NULL; END IF;
    IF p_require_balance AND available_balance(p_account_id) < p_amount THEN RETURN NULL; END IF;
    IF p_stripe_event_id IS NOT NULL AND EXISTS (
        SELECT 1 FROM credit_ledger WHERE stripe_event_id = p_stripe_event_id
    ) THEN RETURN '{}'; END IF;
    FOR v_pool IN
        SELECT purchase_id, expires_at, SUM(delta)::BIGINT AS amount
        FROM credit_ledger WHERE account_id = p_account_id
          AND (expires_at IS NULL OR expires_at > clock_timestamp())
        GROUP BY purchase_id, expires_at HAVING SUM(delta) > 0
        ORDER BY expires_at NULLS LAST, purchase_id NULLS LAST
    LOOP
        EXIT WHEN v_remaining = 0;
        v_take := LEAST(v_remaining, v_pool.amount);
        INSERT INTO credit_ledger(account_id, delta, reason, expires_at, purchase_id,
                                  credential_key_id, credential_credits, memo)
        VALUES (p_account_id, -v_take, p_reason, v_pool.expires_at, v_pool.purchase_id,
                p_credential_key_id, CASE WHEN p_reason = 'credential_issuance' THEN v_take END, p_memo)
        RETURNING id INTO v_id;
        v_ids := v_ids || v_id;
        v_remaining := v_remaining - v_take;
    END LOOP;
    IF v_remaining > 0 THEN
        IF p_require_balance THEN RAISE EXCEPTION 'balance allocation invariant'; END IF;
        INSERT INTO credit_ledger(account_id, delta, reason, memo)
        VALUES (p_account_id, -v_remaining, 'manual_adjustment', p_memo) RETURNING id INTO v_id;
        v_ids := v_ids || v_id;
    END IF;
    RETURN v_ids;
END;
$$;

-- Debt is money, in its original currency; never an at-par credit overdraft.
-- The negative equivalent is only a display when no positive credits remain.
CREATE OR REPLACE FUNCTION available_balance(p_account_id UUID)
RETURNS BIGINT LANGUAGE SQL STABLE AS $$
    WITH balance AS (
        SELECT COALESCE(SUM(delta), 0)::BIGINT AS credits FROM credit_ledger
        WHERE account_id = p_account_id AND (expires_at IS NULL OR expires_at > now())
    ), debt AS (
        SELECT COALESCE(SUM(CEIL(d.debt_minor * p.credits / NULLIF(p.paid_minor, 0))), 0)::BIGINT AS credits
        FROM billing_dispute d JOIN billing_purchase p ON p.id = d.purchase_id
        WHERE p.account_id = p_account_id
    ) SELECT CASE WHEN balance.credits > 0 THEN balance.credits
                  ELSE balance.credits - debt.credits END FROM balance, debt
$$;

-- Consume an outstanding cash debt using credits at each lot's own price.
-- Integer credits are rounded down: we never take extra credits to round cash.
CREATE FUNCTION billing_collect_debt(p_dispute TEXT)
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE
    d RECORD;
    lot RECORD;
    take BIGINT;
    value NUMERIC;
    entry UUID;
BEGIN
    SELECT bd.*, p.account_id, p.currency INTO STRICT d
    FROM billing_dispute bd JOIN billing_purchase p ON p.id = bd.purchase_id WHERE bd.id = p_dispute;
    PERFORM 1 FROM account WHERE id = d.account_id FOR UPDATE;
    FOR lot IN
        SELECT p.*, COALESCE(SUM(cl.delta), 0)::BIGINT AS remaining
        FROM billing_purchase p JOIN credit_ledger cl ON cl.purchase_id = p.id
        WHERE p.account_id = d.account_id AND p.currency = d.currency AND p.paid_minor > 0
          AND (p.expires_at > COALESCE((SELECT billing_closed_at FROM account WHERE id = d.account_id), clock_timestamp())
               OR p.purchased_at >= (SELECT billing_closed_at FROM account WHERE id = d.account_id))
        GROUP BY p.id HAVING SUM(cl.delta) > 0
        ORDER BY (p.id = d.purchase_id) DESC, p.expires_at, p.id
    LOOP
        EXIT WHEN d.debt_minor <= 0;
        take := LEAST(lot.remaining, FLOOR(d.debt_minor * lot.credits / lot.paid_minor)::BIGINT);
        CONTINUE WHEN take <= 0;
        value := take::NUMERIC * lot.paid_minor / lot.credits;
        INSERT INTO credit_ledger(account_id, delta, reason, expires_at, purchase_id, billing_operation_id)
        VALUES (d.account_id, -take, 'dispute_clawback', lot.expires_at, lot.id, d.operation_id)
        RETURNING id INTO entry;
        INSERT INTO billing_dispute_effect(ledger_id, dispute_id, purchase_id, credits, cash_minor)
        VALUES (entry, d.id, lot.id, take, value);
        d.debt_minor := GREATEST(0, d.debt_minor - value);
    END LOOP;
    UPDATE billing_dispute SET debt_minor = d.debt_minor WHERE id = d.id;
END;
$$;

-- Move toward the authoritative cash target, reversing recorded allocations
-- (not today's price or expiry) when a refund overlaps or a dispute is won.
CREATE FUNCTION billing_set_dispute_target(p_dispute TEXT, p_target NUMERIC)
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE
    d RECORD;
    effect RECORD;
    release NUMERIC;
    amount NUMERIC;
    restore BIGINT;
BEGIN
    SELECT bd.*, p.account_id INTO STRICT d FROM billing_dispute bd
    JOIN billing_purchase p ON p.id = bd.purchase_id WHERE bd.id = p_dispute;
    PERFORM 1 FROM account WHERE id = d.account_id FOR UPDATE;
    IF p_target > d.target_minor THEN
        UPDATE billing_dispute SET debt_minor = debt_minor + p_target - target_minor,
            target_minor = p_target WHERE id = d.id;
        PERFORM billing_collect_debt(d.id);
        RETURN;
    END IF;
    release := d.target_minor - p_target;
    amount := LEAST(release, d.debt_minor);
    d.debt_minor := d.debt_minor - amount;
    release := release - amount;
    FOR effect IN
        SELECT e.*, cl.expires_at FROM billing_dispute_effect e
        JOIN credit_ledger cl ON cl.id = e.ledger_id
        WHERE e.dispute_id = d.id AND e.restored_credits < e.credits
        ORDER BY cl.created_at DESC, e.ledger_id
    LOOP
        EXIT WHEN release <= 0 AND p_target <> 0;
        restore := CASE WHEN p_target = 0 THEN effect.credits - effect.restored_credits
                    ELSE LEAST(effect.credits - effect.restored_credits,
                               CEIL(release * effect.credits / effect.cash_minor)::BIGINT) END;
        CONTINUE WHEN restore <= 0;
        amount := restore::NUMERIC * effect.cash_minor / effect.credits;
        INSERT INTO credit_ledger(account_id, delta, reason, expires_at, purchase_id, billing_operation_id)
        VALUES (d.account_id, restore, 'dispute_reversal', effect.expires_at, effect.purchase_id, d.operation_id);
        UPDATE billing_dispute_effect SET restored_credits = restored_credits + restore WHERE ledger_id = effect.ledger_id;
        release := release - amount;
    END LOOP;
    UPDATE billing_dispute SET target_minor = p_target,
        debt_minor = CASE WHEN p_target = 0 THEN 0 ELSE d.debt_minor + GREATEST(0, -release) END
    WHERE id = d.id;
END;
$$;

CREATE FUNCTION billing_refresh_charge(p_purchase TEXT)
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE p RECORD; d RECORD; remaining NUMERIC; target NUMERIC;
BEGIN
    SELECT * INTO STRICT p FROM billing_purchase WHERE id = p_purchase;
    PERFORM 1 FROM account WHERE id = p.account_id FOR UPDATE;
    SELECT GREATEST(0, p.paid_minor
        - COALESCE((SELECT SUM(amount_minor) FROM billing_refund_observation WHERE charge_id = p.charge_id
                    AND status = 'succeeded'), 0)
        - COALESCE((SELECT SUM(r.amount_minor) FROM billing_external_refund e
                    JOIN billing_refund_operation r ON r.id = e.operation_id WHERE r.purchase_id = p.id), 0))
    INTO remaining;
    FOR d IN SELECT * FROM billing_dispute WHERE purchase_id = p.id ORDER BY id LOOP
        target := CASE WHEN d.status IN ('needs_response', 'under_review', 'lost')
                       THEN LEAST(d.amount_minor, remaining) ELSE 0 END;
        PERFORM billing_set_dispute_target(d.id, target);
        remaining := remaining - target;
    END LOOP;
    -- A win can restore paid credits another outstanding dispute is owed.
    -- Settle those debts before any restored credit becomes convertible.
    FOR d IN SELECT bd.id FROM billing_dispute bd JOIN billing_purchase bp ON bp.id = bd.purchase_id
             WHERE bp.account_id = p.account_id AND bd.debt_minor > 0 ORDER BY bd.id LOOP
        PERFORM billing_collect_debt(d.id);
    END LOOP;
END;
$$;
