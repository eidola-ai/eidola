//! The credit arithmetic of a paid request: what it holds, and what it is
//! charged.
//!
//! A request is paid for up front by a **hold**, the most it can cost, and
//! settled at a **charge**, what it did cost; the difference is refunded. The
//! client computes the hold it presents and the gateway recomputes it from the
//! same request and refuses any other amount, so both sides call the functions
//! here and cannot round or order a term differently. Everything is integer
//! arithmetic on scaled rates.
//!
//! # The terms
//!
//! - **The request fee** ([`REQUEST_FEE_CREDITS`]): a fixed amount charged for
//!   every request whose credentials were accepted, whatever became of it. A
//!   request refused after that point (an unknown model, a hold that is not
//!   the worst case, an engine that refuses or fails before answering) is
//!   charged the fee and nothing else ([`refused_charge`]).
//! - **The prompt**, priced one of two ways ([`Prompt`]):
//!   - *counted*: the shared byte contract's chargeable tokens
//!     ([`crate::chargeable_prompt_tokens`]) at the prompt rate, charged at
//!     the engine's reported count clamped to that number;
//!   - *declared*: the client states how many of the prompt's leading tokens
//!     an engine holds in its prefix cache and how many more the prompt may
//!     run to ([`crate::engine_protocol::CachedPrefix`]). The cached tokens
//!     are priced at the cached rate and the rest at the prompt rate. The
//!     engine refuses the request before computing anything unless both
//!     numbers hold, so the price is a function of the request alone and
//!     never of what the cache happened to contain.
//! - **The completion**, priced at its ceiling in the hold and at the reported
//!   count, clamped to the ceiling, in the charge.
//!
//! Each priced term is rounded up to a whole credit by itself, in the hold and
//! in the charge alike, so `charge ≤ hold` term by term.

/// Credits charged for every request whose credentials were accepted: one
/// tenth of a cent. It prices the work a request causes whatever its outcome
/// (verifying its proofs, recording its spends, reading and tokenizing its
/// body, looking up its cached prefix), so a request that is refused is not
/// free.
pub const REQUEST_FEE_CREDITS: u64 = 1_000;

/// The denominator of a scaled per-token rate: a rate of `v` is `v / RATE_SCALE`
/// credits per token.
pub const RATE_SCALE: u64 = 1_000_000;

/// A model's per-token rates, each scaled by [`RATE_SCALE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenRates {
    /// A prompt token the engine computes.
    pub prompt: u64,
    /// A prompt token declared cached, where the model sells one. `None` means
    /// a declared prompt cannot be priced for this model.
    pub cached_prompt: Option<u64>,
    /// A completion token.
    pub completion: u64,
}

/// How a request's prompt is priced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prompt {
    /// By the byte contract: at most `chargeable_tokens`
    /// ([`crate::chargeable_prompt_tokens`]) at the prompt rate.
    Counted { chargeable_tokens: u64 },
    /// By the client's declaration: `cached_tokens` at the cached rate and at
    /// most `new_tokens` more at the prompt rate.
    Declared { cached_tokens: u64, new_tokens: u64 },
}

impl Prompt {
    /// The most prompt tokens this pricing covers. A prompt reported longer is
    /// charged as this many; a declared prompt longer than this is refused by
    /// the engine before it runs.
    pub const fn token_limit(self) -> u64 {
        match self {
            Prompt::Counted { chargeable_tokens } => chargeable_tokens,
            Prompt::Declared {
                cached_tokens,
                new_tokens,
            } => cached_tokens.saturating_add(new_tokens),
        }
    }

    /// Credits for a prompt the engine reported as `prompt_tokens` long, or
    /// `None` for a declared prompt under rates with no cached rate.
    fn credits(self, rates: &TokenRates, prompt_tokens: u64) -> Option<u128> {
        match self {
            Prompt::Counted { chargeable_tokens } => {
                Some(term(prompt_tokens.min(chargeable_tokens), rates.prompt))
            }
            Prompt::Declared {
                cached_tokens,
                new_tokens,
            } => {
                let computed = prompt_tokens.saturating_sub(cached_tokens).min(new_tokens);
                Some(term(cached_tokens, rates.cached_prompt?) + term(computed, rates.prompt))
            }
        }
    }
}

/// One priced term: `tokens` at a scaled `rate`, rounded up to a whole credit.
fn term(tokens: u64, rate: u64) -> u128 {
    (tokens as u128 * rate as u128).div_ceil(RATE_SCALE as u128)
}

/// The hold of a request priced per token: the fee, the prompt at its limit,
/// and the completion at its ceiling. `None` for a declared prompt under rates
/// with no cached rate.
pub fn token_hold(rates: &TokenRates, prompt: Prompt, max_completion_tokens: u64) -> Option<u128> {
    Some(
        REQUEST_FEE_CREDITS as u128
            + prompt.credits(rates, prompt.token_limit())?
            + term(max_completion_tokens, rates.completion),
    )
}

/// The charge of a request priced per token that an engine answered: the fee,
/// the prompt as reported, and the completion as reported, each clamped to
/// what the hold priced. `None` for a declared prompt under rates with no
/// cached rate.
///
/// A declared prompt is charged its whole cached part even when the report is
/// shorter: the engine admitted it only because that many tokens were cached.
pub fn token_charge(
    rates: &TokenRates,
    prompt: Prompt,
    max_completion_tokens: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> Option<u128> {
    Some(
        REQUEST_FEE_CREDITS as u128
            + prompt.credits(rates, prompt_tokens)?
            + term(
                completion_tokens.min(max_completion_tokens),
                rates.completion,
            ),
    )
}

/// The hold, and the charge, of a request to a model priced per request: the
/// fee and the flat price.
pub fn request_hold(per_request_credits: u128) -> u128 {
    REQUEST_FEE_CREDITS as u128 + per_request_credits
}

/// The charge of a request that was refused, or failed, after its credentials
/// were accepted and before an engine produced anything: the fee, or the whole
/// hold where the hold presented is smaller than the fee.
pub fn refused_charge(hold: u128) -> u128 {
    hold.min(REQUEST_FEE_CREDITS as u128)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MiMo-V2.6-Flash as its catalog row prices it, with a cached rate a
    /// quarter of the prompt rate.
    const RATES: TokenRates = TokenRates {
        prompt: 750_000,
        cached_prompt: Some(187_500),
        completion: 3_000_000,
    };
    const FEE: u128 = REQUEST_FEE_CREDITS as u128;

    #[test]
    fn a_counted_hold_is_the_fee_the_chargeable_prompt_and_the_ceiling() {
        let prompt = Prompt::Counted {
            chargeable_tokens: 1_000,
        };
        assert_eq!(token_hold(&RATES, prompt, 8_192), Some(FEE + 750 + 24_576));
    }

    #[test]
    fn a_declared_hold_prices_the_cached_part_at_the_cached_rate() {
        let prompt = Prompt::Declared {
            cached_tokens: 100_096,
            new_tokens: 1_000,
        };
        // 100,096 × 0.1875 = 18,768; 1,000 × 0.75 = 750; 8,192 × 3 = 24,576.
        assert_eq!(
            token_hold(&RATES, prompt, 8_192),
            Some(FEE + 18_768 + 750 + 24_576)
        );
        assert_eq!(prompt.token_limit(), 101_096);
    }

    #[test]
    fn a_declared_prompt_cannot_be_priced_without_a_cached_rate() {
        let rates = TokenRates {
            cached_prompt: None,
            ..RATES
        };
        let prompt = Prompt::Declared {
            cached_tokens: 256,
            new_tokens: 10,
        };
        assert_eq!(token_hold(&rates, prompt, 1), None);
        assert_eq!(token_charge(&rates, prompt, 1, 266, 1), None);
        // A counted prompt needs none.
        let counted = Prompt::Counted {
            chargeable_tokens: 10,
        };
        assert!(token_hold(&rates, counted, 1).is_some());
    }

    #[test]
    fn a_counted_charge_is_clamped_to_the_contract() {
        let prompt = Prompt::Counted {
            chargeable_tokens: 1_000,
        };
        // Reported under the limit: charged as reported.
        assert_eq!(
            token_charge(&RATES, prompt, 8_192, 400, 100),
            Some(FEE + 300 + 300)
        );
        // Reported over either limit: charged at the limit, which is the hold.
        assert_eq!(
            token_charge(&RATES, prompt, 8_192, 5_000, 99_999),
            token_hold(&RATES, prompt, 8_192)
        );
    }

    #[test]
    fn a_declared_charge_is_the_whole_cached_part_and_the_computed_rest() {
        let prompt = Prompt::Declared {
            cached_tokens: 100_096,
            new_tokens: 1_000,
        };
        // 100,500 reported: 404 computed beyond the declared prefix.
        assert_eq!(
            token_charge(&RATES, prompt, 8_192, 100_500, 0),
            Some(FEE + 18_768 + 303)
        );
        // A report shorter than the declared prefix still pays for the prefix.
        assert_eq!(
            token_charge(&RATES, prompt, 8_192, 50_000, 0),
            Some(FEE + 18_768)
        );
        // A report past the limit is charged at the limit.
        assert_eq!(
            token_charge(&RATES, prompt, 8_192, 900_000, 8_192),
            token_hold(&RATES, prompt, 8_192)
        );
    }

    #[test]
    fn no_charge_exceeds_its_hold() {
        let prompts = [
            Prompt::Counted {
                chargeable_tokens: 0,
            },
            Prompt::Counted {
                chargeable_tokens: 12_345,
            },
            Prompt::Declared {
                cached_tokens: 256,
                new_tokens: 0,
            },
            Prompt::Declared {
                cached_tokens: 999_936,
                new_tokens: 4_321,
            },
        ];
        // Rates that leave a remainder in every term, so each rounds.
        let rates = TokenRates {
            prompt: 750_001,
            cached_prompt: Some(187_499),
            completion: 2_999_999,
        };
        for prompt in prompts {
            for ceiling in [1, 8_192, 65_536] {
                let hold = token_hold(&rates, prompt, ceiling).unwrap();
                for reported_prompt in [0, 1, 255, 256, 12_345, 1_000_000, u64::MAX] {
                    for reported_completion in [0, 1, ceiling, ceiling + 1, u64::MAX] {
                        let charge = token_charge(
                            &rates,
                            prompt,
                            ceiling,
                            reported_prompt,
                            reported_completion,
                        )
                        .unwrap();
                        assert!(charge <= hold, "{prompt:?} {ceiling}");
                        assert!(charge >= FEE);
                    }
                }
            }
        }
    }

    #[test]
    fn extreme_inputs_do_not_overflow() {
        let rates = TokenRates {
            prompt: u64::MAX,
            cached_prompt: Some(u64::MAX),
            completion: u64::MAX,
        };
        let prompt = Prompt::Declared {
            cached_tokens: u64::MAX,
            new_tokens: u64::MAX,
        };
        assert_eq!(prompt.token_limit(), u64::MAX);
        let hold = token_hold(&rates, prompt, u64::MAX).unwrap();
        assert!(hold > 0);
    }

    #[test]
    fn a_refused_request_is_charged_the_fee_or_all_it_presented() {
        assert_eq!(refused_charge(50_000), FEE);
        assert_eq!(refused_charge(FEE), FEE);
        assert_eq!(refused_charge(7), 7);
        assert_eq!(refused_charge(0), 0);
        assert_eq!(request_hold(10_000), FEE + 10_000);
    }
}
