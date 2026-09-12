**Summary:** The cart total does not update when an item is added to the cart.

**Severity:** high &nbsp; **Category:** app_bug

## Description
Adding an item to the cart correctly increments the item count but never updates the displayed cart total, leaving shoppers unable to see how much they will be charged.

## Expected Behavior
After adding an item to the cart, the cart total should display the item's price.

## Actual Behavior
The cart total remains blank after the item is added, even though the item count increments correctly.

## Steps to Reproduce
1. Go to the product page
2. Add the item to the cart

_Filed automatically by Sentinel-QA from run `run-golden-001`._
