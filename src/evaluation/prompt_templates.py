"""
Prompt Templates for Zero-Shot and Prompt-Engineered Baselines

This module contains carefully crafted prompts for evaluating base Qwen3
without fine-tuning. Customize the examples in get_prompt_with_examples()
using your best training data examples.
"""


def get_zero_shot_prompt(tweet: str) -> str:
    """
    Minimal zero-shot prompt.
    
    This is the control baseline - just asks for a reply with no guidance.
    
    Args:
        tweet: The tweet to reply to
        
    Returns:
        Formatted prompt string
    """
    return f"Generate an engaging Twitter reply to this tweet:\n\n{tweet}\n\nReply:"


def get_prompt_engineered(tweet: str) -> str:
    """
    Detailed prompt engineering with best practices.
    
    Includes specific instructions about what makes a good reply
    without providing examples.
    
    Args:
        tweet: The tweet to reply to
        
    Returns:
        Formatted prompt string
    """
    prompt = """You are an expert at writing engaging Twitter replies that get likes and retweets.

Guidelines for writing great replies:
- Be concise (under 280 characters)
- Add genuine value or insight to the conversation
- Use natural, conversational language
- Be helpful, thoughtful, or add appropriate humor
- Show expertise or personal experience when relevant
- Ask engaging follow-up questions when appropriate
- Avoid generic responses like "Great post!" or "I agree"
- Avoid promotional content or spam

Tweet to reply to:
{tweet}

Write an engaging reply that follows these guidelines:"""
    
    return prompt.format(tweet=tweet)


def get_prompt_with_examples(tweet: str) -> str:
    """
    Few-shot prompting with hardcoded high-quality examples.
    
    ⚠️ CUSTOMIZE THIS: Add your best tweet-reply pairs from training data!
    
    To customize:
    1. Review data/processed/training_data.jsonl
    2. Pick 5-7 best high-engagement examples
    3. Replace the examples below
    
    Args:
        tweet: The tweet to reply to
        
    Returns:
        Formatted prompt string with examples
    """
    prompt = """You are an expert at writing engaging Twitter replies. Here are examples of great replies:

Example 1:
Tweet: "Just spent 3 hours debugging a React hook that was missing a dependency. fml"
Reply: "useEffect dependency arrays are the modern equivalent of semicolons in JavaScript - you don't think about them until they ruin your day. What was the culprit?"

Example 2:
Tweet: "Hot take: code reviews should focus on architecture, not syntax"
Reply: "Agreed, but syntax discussions catch attention drift. If someone's nitpicking formatting, they might have missed bigger issues. I use it as a canary signal."

Example 3:
Tweet: "Finally hit 10k followers! Thanks everyone 🙏"
Reply: "Congrats! What's been your most effective content strategy? Always looking to learn from folks who've built engaged audiences."

Example 4:
Tweet: "Why does my code work on Friday but break on Monday?"
Reply: "Schrödinger's bug - it exists in a superposition of working and broken until you observe it on Monday morning. (Real answer: probably a race condition or external API change)"

Example 5:
Tweet: "Thinking about switching from JavaScript to TypeScript. Worth it?"
Reply: "Made the switch 2 years ago. The upfront cost is real, but catches so many bugs before runtime. Start with strict:false and gradually tighten. Don't try to convert everything at once."

Now generate an engaging reply to this tweet:
{tweet}

Your reply:"""
    
    return prompt.format(tweet=tweet)


def get_prompt_helpful_style(tweet: str) -> str:
    """
    Prompt optimized for helpful, informative replies.
    
    Args:
        tweet: The tweet to reply to
        
    Returns:
        Formatted prompt string
    """
    prompt = """You are writing a helpful Twitter reply that adds value to the conversation.

Your reply should:
- Provide actionable insights or information
- Share relevant experience or knowledge
- Help solve problems or answer questions
- Be genuinely useful to the reader

Tweet: {tweet}

Write a helpful reply:"""
    
    return prompt.format(tweet=tweet)


def get_prompt_conversational_style(tweet: str) -> str:
    """
    Prompt optimized for engaging, conversational replies.
    
    Args:
        tweet: The tweet to reply to
        
    Returns:
        Formatted prompt string
    """
    prompt = """You are writing a conversational Twitter reply that encourages discussion.

Your reply should:
- Be friendly and approachable
- Ask thought-provoking questions
- Build on the original point
- Encourage back-and-forth dialogue

Tweet: {tweet}

Write an engaging, conversational reply:"""
    
    return prompt.format(tweet=tweet)


def get_prompt_technical_style(tweet: str) -> str:
    """
    Prompt optimized for technical, in-depth replies.
    
    Args:
        tweet: The tweet to reply to
        
    Returns:
        Formatted prompt string
    """
    prompt = """You are writing a technical Twitter reply that demonstrates expertise.

Your reply should:
- Show technical depth and understanding
- Provide specific, actionable details
- Reference best practices or patterns
- Be precise and accurate

Tweet: {tweet}

Write a technically insightful reply:"""
    
    return prompt.format(tweet=tweet)


# Dictionary of all available prompt variants
PROMPT_VARIANTS = {
    "zero_shot": get_zero_shot_prompt,
    "engineered": get_prompt_engineered,
    "with_examples": get_prompt_with_examples,
    "helpful": get_prompt_helpful_style,
    "conversational": get_prompt_conversational_style,
    "technical": get_prompt_technical_style,
}


def get_prompt_description(variant_name: str) -> str:
    """Get description of a prompt variant."""
    descriptions = {
        "zero_shot": "Minimal prompt with no guidance",
        "engineered": "Detailed instructions about good replies",
        "with_examples": "5 hardcoded high-quality examples",
        "helpful": "Optimized for helpful, informative replies",
        "conversational": "Optimized for engaging discussion",
        "technical": "Optimized for technical expertise",
    }
    return descriptions.get(variant_name, "Unknown variant")

