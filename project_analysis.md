# Customer Churn Prediction Project - Initial Analysis

## Problem Analysis

### Business Problem
- **Objective**: Predict customer churn (subscription cancellation) for a music streaming service
- **Data Source**: Event logs from user interactions with the platform
- **Target**: Binary classification - will a user cancel their subscription?

### Technical Challenges Identified

#### 1. Data Quality Issues
- **Class Imbalance**: Churn events are typically rare (5-20% of users), creating severely imbalanced datasets
- **Churn Definition Ambiguity**: Need to define what constitutes "churn" - immediate cancellation vs. inactivity period
- **Data Leakage Risk**: Must avoid using future information to predict past events

#### 2. Feature Engineering Complexity
- **Event Log Structure**: Raw data consists of timestamped user interactions
- **Temporal Patterns**: Need to capture user behavior changes over time
- **Behavioral Signals**: Extract meaningful features from music listening patterns

#### 3. Scale and Performance Requirements
- **Real-time Predictions**: API must handle production traffic
- **Model Drift**: User behavior and music preferences change over time
- **Monitoring**: Need to detect when model performance degrades

## Data Analysis

### Data Structure Overview
From examining `customer_churn_mini.json`, the data contains:

```json
{
  "ts": 1538352117000,           // Timestamp (Unix)
  "userId": "30",                // User identifier
  "sessionId": 29,               // Session identifier
  "page": "NextSong",            // User action type
  "auth": "Logged In",           // Authentication status
  "method": "PUT",               // HTTP method
  "status": 200,                 // Response status
  "level": "paid",               // Subscription tier (paid/free)
  "itemInSession": 50,           // Item number in session
  "location": "Bakersfield, CA", // User location
  "userAgent": "Mozilla/5.0...", // Browser info
  "lastName": "Freeman",         // User last name
  "firstName": "Colin",          // User first name
  "registration": 1538173362000, // Registration timestamp
  "gender": "M",                 // User gender
  "artist": "Martha Tilston",    // Artist name (if music)
  "song": "Rockpools",          // Song name (if music)
  "length": 277.89016           // Song length (if music)
}
```

### Key Event Types Observed
- `NextSong`: User plays next song
- `Add to Playlist`: User adds song to playlist
- `Thumbs Up/Down`: User rates songs
- `Roll Advert`: Advertisement shown (free users)
- `Downgrade`: User downgrades subscription

### Critical Data Insights

#### 1. Churn Indicators
- **Downgrade Events**: Direct signal of subscription changes
- **Reduced Activity**: Fewer NextSong events over time
- **Engagement Drops**: Less playlist creation, rating activity
- **Session Patterns**: Shorter sessions, longer gaps between usage

#### 2. User Segmentation
- **Subscription Tiers**: Paid vs Free users have different behavior patterns
- **Geographic Distribution**: Location may influence churn risk
- **Engagement Level**: Power users vs casual listeners

#### 3. Temporal Patterns
- **Registration Recency**: New users may have different churn patterns
- **Usage Consistency**: Regular vs sporadic usage patterns
- **Time-of-day/week Preferences**: Activity timing patterns
