//! Shared, backend-free post fixtures for the chat/space render surface.
//!
//! These build `PostNode` trees (the `get_space_tree` render DTO) directly,
//! with no `Core` and no HTTP, so the chat surface can be exercised both by the
//! visual snapshot cases and by the `eidola_space` example. Depends only on
//! public `eidola_app_core` types — keep it that way so the example (which
//! links the library, not the test tree) can `#[path]`-include it.

use eidola_app_core::{PostBlock, PostNode, PostParticipant, PostReference};

/// Build a fixture `PostNode` for the branch/generation snapshot cases. The
/// flattener's depth/branch metadata is supplied directly so the render can be
/// exercised without a backend.
#[allow(clippy::too_many_arguments)]
pub fn fixture_post(
    action_id: &str,
    kind: &str,
    label: &str,
    action_type: &str,
    text: &str,
    depth: usize,
    is_branch: bool,
    generation_count: i64,
) -> PostNode {
    PostNode {
        action_id: action_id.into(),
        item_id: format!("item-{action_id}"),
        parent_action_id: None,
        participant: PostParticipant {
            kind: kind.into(),
            label: label.into(),
        },
        action_type: action_type.into(),
        generation: generation_count - 1,
        generation_count,
        is_current: true,
        model: (kind == "agent").then(|| label.to_string()),
        credits_consumed: (kind == "agent").then_some(700),
        relation: (depth > 0 || is_branch).then(|| "reply".to_string()),
        depth,
        is_branch,
        blocks: vec![PostBlock {
            id: String::new(),
            block_type: "text".into(),
            text: Some(text.into()),
            tool_name: None,
            tool_call_id: None,
            data: None,
        }],
        references: Vec::new(),
        created_at: 0,
    }
}

/// Char-offset range of `phrase` within `content` (for a reference edge's
/// quoted span). Robust to multibyte chars before the phrase.
pub fn char_range(content: &str, phrase: &str) -> (i64, i64) {
    let byte = content
        .find(phrase)
        .unwrap_or_else(|| panic!("reference phrase not found: {phrase:?}"));
    let start = content[..byte].chars().count() as i64;
    (start, start + phrase.chars().count() as i64)
}

/// The "kitchen sink" — a realistic, multi-party space exercising every render
/// element we care about, authored in the flattener's pre-order (spine first,
/// then branches) so it matches what `get_space_tree` would produce. Two humans
/// ("You" + "Mara") and an AI ("kimi-k2") debate Thrasymachus's shepherd from
/// *Republic* I and its bearing on modern paternalism, correcting each other on
/// fact and framing, branching a tangent, quoting one another inline, and
/// collaboratively editing a newsletter excerpt as the product. Used as the
/// non-interactive visual reference for iterating on the chat surface.
pub fn kitchen_sink_posts() -> Vec<PostNode> {
    // Substantive bodies kept in variables so reference spans can be computed
    // against the exact text.
    let a2 = "Three threads worth pulling apart.\n\n\
        First, the position. Thrasymachus's opening definition is that justice is \
        *the advantage of the stronger* (338c): in any regime the ruling party makes \
        laws to its own benefit and calls obedience to them \"just.\" This isn't quite \
        the cartoon of *might makes right* — it's a claim about whose interest the word \
        *justice* actually tracks.\n\n\
        Second, the shepherd. When Socrates answers that every craft seeks the good of \
        its object — medicine the patient's health, not the doctor's — Thrasymachus \
        erupts with the shepherd (343b): shepherds fatten sheep for the feast, not for \
        the sheep, and rulers are no different. The image denies that ruling is a craft \
        aimed at the ruled.\n\n\
        Third, paternalism. The modern \"we know what's good for you\" state is almost \
        the photographic negative of Thrasymachus: the paternalist *adopts* the \
        shepherd's good-of-the-flock language to justify coercion. Thrasymachus would \
        call that the oldest trick — the stronger dressing its advantage in the \
        vocabulary of care.\n\n\
        And yes: Socrates was rattled. Thrasymachus is described bursting in \"like a \
        wild beast,\" and Socrates says he was struck with fear.";

    let a4 = "Fair — I conflated the definition with its defense. Corrected: (1) justice \
        is the advantage of the stronger (338c); (2) Socrates' craft-analogy; (3) *then* \
        the shepherd at 343b, as rebuttal.\n\n\
        Let me also sharpen the \"caught off guard\" point, because it's easy to \
        overstate. Socrates was frightened by his manner, not refuted by his argument — \
        the wild-beast entrance (336b) unsettles him, but he recovers and, by 350d, has \
        forced Thrasymachus into the contradiction that makes him *blush*. So \"caught \
        flat-footed\" is true of the rhetoric and false of the dialectic.";

    let excerpt = "**This week: the shepherd's bargain.** In *Republic* I, Thrasymachus is \
        too often flattened into \"might makes right.\" His sharper move is the shepherd \
        (343b): he denies that ruling is a craft aimed at the good of the ruled, the way \
        medicine aims at health. Modern paternalism runs the image in reverse — it keeps \
        the shepherd's tender vocabulary of care while keeping the shepherd's appetite. \
        The lesson isn't cynicism; it's a question to put to any \"for your own good\": \
        whose advantage does the rule actually track? Even Socrates was frightened by his \
        manner before he answered the argument — force of delivery is not force of reason.";

    // Reference spans, computed against the exact source text.
    let (r3s, r3e) = char_range(a2, "opening definition");
    let (r8s, r8e) = char_range(a4, "frightened by his manner");

    let reply_ref = |antecedent: &str, start: i64, end: i64| PostReference {
        antecedent_action_id: antecedent.into(),
        ordinal: 1,
        content_block_id: None,
        range_start: Some(start),
        range_end: Some(end),
        annotation: None,
        snippet: None,
    };

    // (action_id, item_id, kind, label, type, depth, is_branch, parent, gen, refs, content)
    let post = |action_id: &str,
                kind: &str,
                label: &str,
                atype: &str,
                depth: usize,
                is_branch: bool,
                parent: Option<&str>,
                generation: i64,
                refs: Vec<PostReference>,
                content: &str,
                at: i64|
     -> PostNode {
        let mut n = fixture_post(
            action_id, kind, label, atype, content, depth, is_branch, generation,
        );
        n.parent_action_id = parent.map(String::from);
        n.references = refs;
        n.relation = parent.map(|_| "reply".to_string());
        n.created_at = at;
        n
    };

    vec![
        // Spine.
        post(
            "a1",
            "human",
            "user",
            "user_input",
            0,
            false,
            None,
            1,
            vec![],
            "I keep circling back to Thrasymachus and his shepherd in *Republic* I. The \
             claim, as I remember it, is that rulers tend their subjects the way a shepherd \
             fattens sheep — for the master's table, not the flock's good — and that justice \
             is just a name for whatever serves the stronger. How much of that actually maps \
             onto the modern paternalist state, the \"we know what's good for you\" kind? And \
             is the old story true that Socrates was caught flat-footed by him?",
            1,
        ),
        post(
            "a2",
            "agent",
            "kimi-k2",
            "inference",
            0,
            false,
            Some("a1"),
            1,
            vec![],
            a2,
            2,
        ),
        post(
            "a3",
            "human",
            "Mara",
            "user_input",
            0,
            false,
            Some("a2"),
            1,
            vec![reply_ref("a2", r3s, r3e)],
            "One correction on the order. You called the shepherd part of his *opening \
             definition*, but it isn't — it's his counter-punch. The definition (advantage of \
             the stronger) comes first; the shepherd only appears at 343b, *after* Socrates \
             presses the craft-analogy. The sequence matters: the shepherd is a rebuttal to a \
             specific argument, not a free-standing thesis.",
            3,
        ),
        post(
            "a4",
            "agent",
            "kimi-k2",
            "inference",
            0,
            false,
            Some("a3"),
            1,
            vec![],
            a4,
            4,
        ),
        post(
            "a7",
            "human",
            "Mara",
            "user_input",
            0,
            false,
            Some("a4"),
            1,
            vec![],
            "Good — so the newsletter through-line writes itself: Thrasymachus isn't \"might \
             makes right,\" he's denying that ruling is a craft that serves the ruled, and \
             modern paternalism is the inversion — it borrows the shepherd's good-of-the-flock \
             framing to license power. Let me draft the excerpt.",
            7,
        ),
        // The collaboratively edited node (the "product"): drafted by You,
        // refined by Mara — its current tip is Mara's (cross-author edit), so it
        // reads as a v2 generation authored by Mara.
        post(
            "a8",
            "human",
            "Mara",
            "user_input",
            0,
            false,
            Some("a7"),
            2,
            vec![reply_ref("a4", r8s, r8e)],
            excerpt,
            9,
        ),
        post(
            "a9",
            "agent",
            "kimi-k2",
            "inference",
            0,
            false,
            Some("a8"),
            1,
            vec![],
            "One flag before this ships: the excerpt lets \"advantage of the stronger\" and \
             the shepherd stand as a single move. They're two — the definition (338c) and its \
             defense (343b). A reader who knows the text will catch the seam; otherwise it \
             lands.",
            10,
        ),
        // Branch: a later tangent off a2 (so it indents off the spine).
        post(
            "a5",
            "human",
            "user",
            "user_input",
            1,
            true,
            Some("a2"),
            1,
            vec![],
            "A tangent off your first answer: when you say justice is \"the advantage of the \
             stronger,\" is Thrasymachus making a hard descriptive claim about how power \
             behaves — proto-Machiavelli — or a *definitional* one about what the word means? \
             Those come apart fast.",
            5,
        ),
        post(
            "a6",
            "agent",
            "gemma-4-E2B_q4_0-it@local",
            "inference",
            1,
            false,
            Some("a5"),
            1,
            vec![],
            "Definitional — and that's his vulnerability. If it's merely descriptive (\"the \
             strong usually win\"), it's banal and Socrates has no quarrel. By making it a \
             *definition* of justice, Thrasymachus must defend that the ruler qua ruler never \
             errs about his own advantage, and that's exactly the crack Socrates pries open at \
             340c.",
            6,
        ),
    ]
}

/// The quoted-references scene's source-post body — the text every range in
/// [`quoted_reference_posts`] is measured against. A `fn`, not an inline
/// literal, so a caller (the driver's composing scene) computes the same byte
/// offsets rather than hard-coding them past a multi-byte em dash.
pub fn quoted_reference_source() -> String {
    "The shepherd image is doing more work than it looks. Thrasymachus is not saying rulers \
     are cruel; he is saying that ruling, like shepherding, is a craft whose end lies outside \
     its object. The sheep are fattened, tended, protected — and none of that is for the \
     sheep.\n\n\
     What makes it hard to answer is that the care is real. A paternalist state that genuinely \
     improves lives is not thereby refuted; the question Thrasymachus forces is whose advantage \
     the rule tracks when the two come apart."
        .to_string()
}

/// The passage the composing scene quotes ("the care is real"), as a byte
/// range into [`quoted_reference_source`].
pub fn quoted_reference_selection() -> std::ops::Range<usize> {
    let src = quoted_reference_source();
    let (s, e) = byte_range(&src, "the care is real");
    s as usize..e as usize
}

/// Byte offsets of `phrase` within `content` (the schema's units, unlike the
/// char-based [`char_range`] the older fixtures use).
pub fn byte_range(content: &str, phrase: &str) -> (i64, i64) {
    let start = content
        .find(phrase)
        .unwrap_or_else(|| panic!("reference phrase not found: {phrase:?}"));
    (start as i64, (start + phrase.len()) as i64)
}

/// One incoming reference in [`quoted_reference_posts`] — the shape the scene
/// converts into `eidola_app_core::IncomingReference` when seeding a `Space`.
pub struct QuotedIncoming {
    pub action_id: String,
    pub block_id: String,
    pub range: (i64, i64),
}

/// The quoted-references scene: a source post whose passages other posts have
/// quoted, and the replies that quote them — the fixture behind the wave-2
/// footnote rail, embed blocks, and source highlights.
///
/// Returns `(posts, incoming)` where `incoming` is the reverse index the
/// source post's highlights are painted from (`(quoted action, references)`).
/// Ranges are **byte** offsets into the source block's text, exactly as the
/// schema stores them.
#[allow(clippy::type_complexity)]
pub fn quoted_reference_posts() -> (Vec<PostNode>, Vec<(String, Vec<QuotedIncoming>)>) {
    let source = quoted_reference_source();
    let src = source.as_str();

    let quoted_a = "the care is real";
    let quoted_b = "whose advantage the rule tracks";
    let (a_start, a_end) = byte_range(src, quoted_a);
    let (b_start, b_end) = byte_range(src, quoted_b);
    // The branch quotes a longer span starting at the same point — an overlap
    // with the reply's, so a click on the shared text is ambiguous (the picker).
    let a_long_end = a_end + 30;

    let block = "blk-1";
    let mut source_post = fixture_post("q1", "agent", "kimi-k2", "inference", src, 0, false, 1);
    source_post.blocks[0].id = block.into();
    source_post.created_at = 1;

    // A reply quoting the first passage: the marker stands as its own
    // paragraph (what the editor renders as a quote block) with prose around it.
    let mut reply = fixture_post(
        "q2",
        "human",
        "user",
        "user_input",
        "That's the sentence I keep snagging on:\n\n{{ embed 1 }}\n\nIf the care is real, \
         doesn't the shepherd analogy quietly concede Socrates' point — that the craft does \
         aim at its object after all?",
        0,
        false,
        1,
    );
    reply.parent_action_id = Some("q1".into());
    reply.relation = Some("reply".into());
    reply.created_at = 2;
    reply.references = vec![PostReference {
        antecedent_action_id: "q1".into(),
        ordinal: 1,
        content_block_id: Some(block.into()),
        range_start: Some(a_start),
        range_end: Some(a_end),
        annotation: None,
        snippet: Some(quoted_a.into()),
    }];

    // A branch quoting the overlapping span plus a second passage — two
    // references on one post (the rail's plural case).
    let mut branch = fixture_post(
        "q3",
        "human",
        "Mara",
        "user_input",
        "Two things, separately.\n\n{{ embed 1 }}\n\nand\n\n{{ embed 2 }}\n\nThe first is a \
         concession; the second is the actual test. I'd keep the second and drop the first.",
        1,
        true,
        1,
    );
    branch.parent_action_id = Some("q1".into());
    branch.relation = Some("reply".into());
    branch.created_at = 3;
    branch.references = vec![
        PostReference {
            antecedent_action_id: "q1".into(),
            ordinal: 1,
            content_block_id: Some(block.into()),
            range_start: Some(a_start),
            range_end: Some(a_long_end),
            annotation: None,
            snippet: src
                .get(a_start as usize..a_long_end as usize)
                .map(String::from),
        },
        PostReference {
            antecedent_action_id: "q1".into(),
            ordinal: 2,
            content_block_id: Some(block.into()),
            range_start: Some(b_start),
            range_end: Some(b_end),
            annotation: None,
            snippet: Some(quoted_b.into()),
        },
    ];

    let incoming = vec![(
        "q1".to_string(),
        vec![
            QuotedIncoming {
                action_id: "q2".into(),
                block_id: block.into(),
                range: (a_start, a_end),
            },
            QuotedIncoming {
                action_id: "q3".into(),
                block_id: block.into(),
                range: (a_start, a_long_end),
            },
            QuotedIncoming {
                action_id: "q3".into(),
                block_id: block.into(),
                range: (b_start, b_end),
            },
        ],
    )];

    (vec![source_post, reply, branch], incoming)
}
