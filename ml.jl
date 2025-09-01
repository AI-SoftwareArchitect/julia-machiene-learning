using Random
using DataFrames
using MLDataUtils
using DecisionTree
using Statistics
using Plots

# --- Mock Data Generation ---
function generate_mock_data(n::Int=500)
    Random.seed!(42)
    x1 = randn(n)
    x2 = randn(n)
    x3 = randn(n)
    # Binary classification target
    y_class = map(x -> x > 0.0 ? 1 : 0, x1 + 0.5*x2 + 0.2*x3 + 0.1*randn(n))
    # Regression target
    y_reg = 2.0*x1 - x2 + 0.5*x3 + randn(n)
    df = DataFrame(x1=x1, x2=x2, x3=x3, y_class=y_class, y_reg=y_reg)
    return df
end

# --- Data Preprocessing ---
function preprocess(df::DataFrame)
    # Train-test split
    train_inds, test_inds = splitobs(1:size(df,1); at=0.7)
    train = df[train_inds, :]
    test = df[test_inds, :]
    return train, test
end

# --- Model Training & Evaluation ---
function train_classification(train, test)
    features = Matrix(select(train, Not([:y_class, :y_reg])))
    labels = train.y_class
    model = DecisionTreeClassifier(max_depth=5)
    fit!(model, features, labels)

    # Test predictions
    test_features = Matrix(select(test, Not([:y_class, :y_reg])))
    preds = predict(model, test_features)

    acc = mean(preds .== test.y_class)
    println("Classification Accuracy: ", acc)
    return model
end

function train_regression(train, test)
    features = Matrix(select(train, Not([:y_class, :y_reg])))
    labels = train.y_reg
    model = DecisionTreeRegressor(max_depth=5)
    fit!(model, features, labels)

    # Test predictions
    test_features = Matrix(select(test, Not([:y_class, :y_reg])))
    preds = predict(model, test_features)

    rmse = sqrt(mean((preds .- test.y_reg).^2))
    println("Regression RMSE: ", rmse)
    return model
end

# --- Main ---
df = generate_mock_data(1000)
train, test = preprocess(df)

clf_model = train_classification(train, test)
reg_model = train_regression(train, test)
