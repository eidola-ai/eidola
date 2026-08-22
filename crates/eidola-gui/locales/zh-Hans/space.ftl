# The conversation window (src/space_view/mod.rs), Simplified Chinese.

space-error-archived = 此对话已归档，无法接收新的回复。
space-error-not-joined = 这是你的助理之间开启的对话。你可以阅读全部内容；要参与则需加入，而此版本尚无法做到。

space-footnote-delegation-concluded = 已告一段落
space-footnote-delegation-paused = 已达到回复上限（{ $depth }/{ $limit }）
space-footnote-delegation-budget = 已用完全部 { $limit } 个回合
space-footnote-delegation-failed = 已中止：{ $reason ->
        [upstream] 无法连接到模型
        [funding] 无法为该回合付费
        [configuration] 其设置存在问题
       *[other] 有一个回合未能完成
    }

space-regenerating = 正在重新生成…
space-error-response-truncated = 模型把全部长度额度都用在了思考上，始终没有开始作答。没有新增或替换任何回答——请重试，或换一个更简短的问题。
space-regenerating-elsewhere = 这条回复正在重新生成。
space-answer-cut-off = 这条回复达到了长度上限，在中途停下。
